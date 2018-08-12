package net.b07z.sepia.nlu.trainers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Logger;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEnt;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.util.MalletLogger;
import net.b07z.sepia.nlu.mallet.CharSequence2TokenSequenceCustom;
import net.b07z.sepia.nlu.tools.CustomDataHandler;

public class MalletIntentTrainer implements IntentTrainer {
	
	private static Logger logger = MalletLogger.getLogger(MalletNerTrainer.class.getName());
	
	Properties props;
	private String modelFile;
	private Reader trainingFile;
	private Classifier classifierModel;
	private String languageCode;
	
	private String defaultLabel = "OTHER";			//Label for initial context and uninteresting tokens

	/**
	 * Setup NER classifier with properties file and training data.
	 * @param propertiesFile
	 * @param trainDataFileBase
	 * @param modelOutputFileBase
	 * @param languageCode
	 * @throws Exception 
	 */
	public MalletIntentTrainer(String propertiesFile, String trainDataFileBase, String modelOutputFileBase, String languageCode) throws Exception {
		this.languageCode = languageCode;
		this.modelFile = modelOutputFileBase + "_" + this.languageCode;
		String trainDataFile = trainDataFileBase + "_" + this.languageCode;
		this.trainingFile = new FileReader(new File(trainDataFile));
		
		if (propertiesFile != null && !propertiesFile.isEmpty()){
			props = CustomDataHandler.loadProperties(propertiesFile);
			if (props.containsKey("defaultLabel")) defaultLabel = props.getProperty("defaultLabel");
			//TODO: ...
			logger.info("Default label: " + defaultLabel);
		}
	}
	
	@Override
	public void train() {
		long tic = System.currentTimeMillis();
		//Begin by importing documents from text to feature sequences
        List<Pipe> pipeList = new ArrayList<Pipe>();

        //Pipes - most of this is already done by our own tokenizer and the stored training data
        /*
        pipeList.add(new CharSequenceLowercase());
        pipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        pipeList.add(new TokenSequenceRemoveStopwords(new File("stoplists/en.txt"), "UTF-8", false, false, false));
        */
        pipeList.add(new Target2Label());
        //pipeList.add(new CharSequence2TokenSequence());
        pipeList.add(new CharSequence2TokenSequenceCustom("[^\\s\\n\\r]+")); 	//NOTE: we only split on whitespace because our tokenizer demands it ([^\\s\\n\\r]+)
        pipeList.add(new TokenSequence2FeatureSequence());
        pipeList.add(new FeatureSequence2FeatureVector());
        SerialPipes pipe = new SerialPipes(pipeList);

        //prepare training instances
        InstanceList trainingInstanceList = new InstanceList(pipe);
        trainingInstanceList.addThruPipe(new CsvIterator(this.trainingFile, "(\\w+)\\s(.*)", 2, 1, -1));	//NOTE: identical to train data export

        ClassifierTrainer<MaxEnt> trainer = new MaxEntTrainer();	//new SVMClassifierTrainer(new LinearKernel());		//new NaiveBayesTrainer();
        this.classifierModel = trainer.train(trainingInstanceList);

        if (this.modelFile != null) {
			try (ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(modelFile))){
				s.writeObject(this.classifierModel);
				
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
}
