package net.b07z.sepia.nlu.trainers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.util.Properties;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByLabelLikelihood;
import cc.mallet.fst.CRFTrainerByThreadedLabelLikelihood;
import cc.mallet.fst.SimpleTagger;
import cc.mallet.fst.Transducer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.util.MalletLogger;
import net.b07z.sepia.nlu.tools.CustomDataHandler;

/**
 * MALLET CRF classifier for training.
 * 
 * @author Florian Quirin
 *
 */
public class MalletNerTrainer implements NerTrainer {
	
	private static Logger logger = MalletLogger.getLogger(MalletNerTrainer.class.getName());
	
	Properties props;
	private String modelFile;
	private Reader trainingFile;
	private CRF crfModel;
	private String languageCode;
	
	private String defaultLabel = "O";			//Label for initial context and uninteresting tokens
	private double gaussianVariance = 10.0;		//The gaussian prior variance used for training
	private int[] orders = {1, 2};				//List of label Markov orders (main and backoff)
	private String forbiddenRegEx = "\\s";		//label1, label2 transition forbidden if it matches this
	private String allowedRegEx = ".*";			//label1, label2 transition allowed only if it matches this
	private boolean connected = true;			//Include all allowed transitions, even those not in training data
	private boolean featureInduction = false; 	//Whether to perform feature induction during training
	private String weigths = "some-dense";		//Use sparse, some-dense (using a heuristic), or dense features on transitions.
	private int iterations = 500;				//Number of training iterations
	private int numThreads = 1;					//Number of threads to use for CRF training

	/**
	 * Setup NER classifier with properties file and training data.
	 * @param propertiesFile
	 * @param trainDataFileBase
	 * @param modelOutputFileBase
	 * @param languageCode
	 * @throws Exception 
	 */
	public MalletNerTrainer(String propertiesFile, String trainDataFileBase, String modelOutputFileBase, String languageCode) throws Exception{
		this.languageCode = languageCode;
		this.modelFile = modelOutputFileBase + "_" + this.languageCode;
		String trainDataFile = trainDataFileBase + "_" + this.languageCode;
		this.trainingFile = new FileReader(new File(trainDataFile));
		
		if (propertiesFile != null && !propertiesFile.isEmpty()){
			props = CustomDataHandler.loadProperties(propertiesFile);
			if (props.containsKey("defaultLabel")) defaultLabel = props.getProperty("defaultLabel");
			//TODO: ...
		}
	}
	
	@Override
	public void train() {
		long tic = System.currentTimeMillis();
		InstanceList trainingData = null;
		
		Pipe p = new SimpleTagger.SimpleTaggerSentence2FeatureVectorSequence();
		p.getTargetAlphabet().lookupIndex(defaultLabel);
		
		p.setTargetProcessing(true);
		trainingData = new InstanceList(p);
		trainingData.addThruPipe(new LineGroupIterator(trainingFile, Pattern.compile("^\\s*$"), true));

		logger.info("Number of features in training data: " + p.getDataAlphabet().size());
		logger.info("Number of predicates: " + p.getDataAlphabet().size());
		//Labels debug:
		Alphabet targets = p.getTargetAlphabet();
		StringBuffer buf = new StringBuffer("Labels (train):");
		for (int i = 0; i < targets.size(); i++){
			buf.append(" ").append(targets.lookupObject(i).toString());
		}
		logger.info(buf.toString());
		
		this.crfModel = null;
		this.crfModel = train(trainingData,
				this.orders, this.defaultLabel,	this.forbiddenRegEx, this.allowedRegEx,
				this.connected, this.iterations, this.gaussianVariance,
				this.numThreads, this.weigths, this.featureInduction,
				this.crfModel);
		
		if (modelFile != null) {
			try (ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(modelFile))){
				s.writeObject(this.crfModel);
				
			}catch (FileNotFoundException e){
				e.printStackTrace();
				throw new RuntimeException("Could not write model file: " + e.getMessage());
			
			}catch (IOException e){
				e.printStackTrace();
				throw new RuntimeException("Could not write model file: " + e.getMessage());
			}
		} 
		try {
			trainingFile.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		logger.info("Training took " + (System.currentTimeMillis() - tic) + "ms.");
	}
	
	//-----------------------------------------------------------------------------------
	
	/**
	 * Create and train a CRF model from the given training data,
	 * optionally testing it on the given test data.
	 *
	 * @param training training data
	 * @param orders label Markov orders (main and backoff)
	 * @param defaultLabel default label
	 * @param forbidden regular expression specifying impossible label
	 * transitions <em>current</em><code>,</code><em>next</em>
	 * (<code>null</code> indicates no forbidden transitions)
	 * @param allowed regular expression specifying allowed label transitions
	 * (<code>null</code> indicates everything is allowed that is not forbidden)
	 * @param connected whether to include even transitions not
	 * occurring in the training data.
	 * @param iterations number of training iterations
	 * @param var Gaussian prior variance
	 * @param numThreads threads to use for training
	 * @param weights use sparse, some-dense (using a heuristic), or dense features on transitions
	 * @param featureInduction whether to perform feature induction during training
	 * @return the trained model
	 */
	public static CRF train(InstanceList training, int[] orders, String defaultLabel,
						String forbidden, String allowed, boolean connected, int iterations, double var,
						int numThreads, String weights, boolean featureInduction, 
						CRF crf) {

		Pattern forbiddenPat = Pattern.compile(forbidden);
		Pattern allowedPat = Pattern.compile(allowed);
		if (crf == null) {
			crf = new CRF(training.getPipe(), (Pipe)null);
			String startName = crf.addOrderNStates(training, orders, null,
						defaultLabel, forbiddenPat, allowedPat,	connected);
			for (int i = 0; i < crf.numStates(); i++){
				crf.getState(i).setInitialWeight (Transducer.IMPOSSIBLE_WEIGHT);
			}
			crf.getState(startName).setInitialWeight(0.0);
		}
		logger.info("Training on " + training.size() + " instances");
 
		if (numThreads > 1){
			//TODO: this should be an interface to avoid redundant code ...
			CRFTrainerByThreadedLabelLikelihood crft = new CRFTrainerByThreadedLabelLikelihood(crf, numThreads);
			crft.setGaussianPriorVariance(var);
   
			if (weights.equals("dense")){
				crft.setUseSparseWeights(false);
				crft.setUseSomeUnsupportedTrick(false);
			
			}else if (weights.equals("some-dense")){
				crft.setUseSparseWeights(true);
				crft.setUseSomeUnsupportedTrick(true);
			
			}else if (weights.equals("sparse")){
				crft.setUseSparseWeights(true);
				crft.setUseSomeUnsupportedTrick(false);
				
			}else{
				throw new RuntimeException("Unknown weights option: " + weights);
			}
   
			if (featureInduction){
				throw new IllegalArgumentException("Multi-threaded feature induction is not yet supported.");
			
			}else{
				boolean converged;
				for (int i = 1; i <= iterations; i++){
					converged = crft.train(training, 1);
					if (converged){
						break;
					}
				}
			}
			crft.shutdown();
		
		}else{
			//TODO: this should be an interface to avoid redundant code ...
			
			/*
			CRFOptimizableByLabelLikelihood optLabel = new CRFOptimizableByLabelLikelihood(crf, training);
			Optimizable.ByGradientValue[] opts = new Optimizable.ByGradientValue[] { optLabel };
			
			CRFTrainerByValueGradients crft = new CRFTrainerByValueGradients(crf, opts);
			crft.setMaxResets(0);
			*/
			
			CRFTrainerByLabelLikelihood crft = new CRFTrainerByLabelLikelihood(crf);
			crft.setGaussianPriorVariance(var);
			   
			if (weights.equals("dense")){
				crft.setUseSparseWeights(false);
				crft.setUseSomeUnsupportedTrick(false);
			
			}else if (weights.equals("some-dense")){
				crft.setUseSparseWeights(true);
				crft.setUseSomeUnsupportedTrick(true);
			
			}else if (weights.equals("sparse")){
				crft.setUseSparseWeights(true);
				crft.setUseSomeUnsupportedTrick(false);
			
			}else{
				throw new RuntimeException("Unknown weights option: " + weights);
			}
   
			if (featureInduction){
				crft.trainWithFeatureInduction(training, null, null, null, iterations, 10, 20, 500, 0.5, false, null);
			}else{
				boolean converged;
				for (int i = 1; i <= iterations; i++){
					converged = crft.train(training, 1);
					if (converged){
						break;
					}
				}
			}
		}
		return crf;
	}

}
