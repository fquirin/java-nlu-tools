package net.b07z.sepia.nlu.classifiers;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.MaxLatticeDefault;
import cc.mallet.fst.Transducer;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Sequence;
import cc.mallet.util.MalletLogger;
import net.b07z.sepia.nlu.tokenizers.RealLifeChatTokenizer;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;

/**
 * MALLET implementation of CRF NER classifier.
 * 
 * @author Florian Quirin
 *
 */
public class MalletNerClassifier implements NerClassifier{
	
	private static Logger logger = MalletLogger.getLogger(MalletNerClassifier.class.getName());
	
	private CRF crfModel;
	private Tokenizer tokenizer;
	private String languageCode;
	
	private int nBest = 3; 				//How many answers to output
	private int cacheSize = 100000; 	//How much state information to memoize in n-best decoding
	
	/**
	 * Create NER classifier with model and tokenizer.
	 * @param modelFileBase
	 * @param tokenizer
	 * @param languageCode
	 */
	public MalletNerClassifier(String modelFileBase, Tokenizer tokenizer, String languageCode) throws Exception {
		this.languageCode = languageCode;
		//load model
		String modelFile = modelFileBase + "_" + this.languageCode;
		try (ObjectInputStream s = new ObjectInputStream(new FileInputStream(modelFile))){
			this.crfModel = (CRF) s.readObject();
		}catch(Exception e){
			throw e;
		}
		this.tokenizer = tokenizer;
	}

	@Override
	public List<NerEntry> analyzeSentence(String sentence) {
		//Normalize and get tokens
		List<String> tokens = tokenizer.getTokens(sentence);
		
		//Create instance - TODO: there has got to be an easier way ...
		String data = String.join("\n", tokens);	//if you want to use System.getProperty("line.separator") check SentenceToFeatureVectorSequencePipe
		Instance carrier = new Instance(data, null, "userinputgroup0", null);
		
		//Get pipe and add instance
		Pipe p = this.crfModel.getInputPipe();
		p.setTargetProcessing(false);
		InstanceList testData = new InstanceList(p);
		testData.addThruPipe(carrier);
		/*
		String testFile = "data/mallet_test2.txt";
		try {
			testData.addThruPipe(new LineGroupIterator(new FileReader(new File(testFile)), Pattern.compile("^\\s*$"), true));
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		*/
		//Labels debug:
		/*
		if (!p.isTargetProcessing()) {
			Alphabet targets = p.getTargetAlphabet();
			StringBuffer buf = new StringBuffer("Labels (model):");
			for (int i = 0; i < targets.size(); i++)
				buf.append(" ").append(targets.lookupObject(i).toString());
			logger.info(buf.toString());
		}
		*/
		
		List<NerEntry> nerEntries = new ArrayList<>();
		boolean includeInput = false;

		Instance inst = testData.get(0);
		/*
		System.out.println("Data: " + inst.getData());
		System.out.println("Target: " + inst.getTarget());
		System.out.println("Name: " + inst.getName());
		System.out.println("Source: " + inst.getSource());
		//System.out.println("Labeling: " + inst.getLabeling());
		*/
		Sequence<?> input = (Sequence<?>) inst.getData();
		Sequence<?>[] outputs = apply(this.crfModel, input, nBest);
		int k = outputs.length;
		for (int j = 0; j < input.size(); j++){
			String token = tokens.get(j);
			if (includeInput){
				FeatureVector fv = (FeatureVector) input.get(j);
				logger.fine("j: " + j + " - " + fv.toString(true));                
			}
			String bestLabel = "";
			List<TokenLabel> nLabels = new ArrayList<>();
			for (int a = 0; a < k; a++){
				String label = outputs[a].get(j).toString();
				if (a == 0){
					bestLabel = label;
				}
				nLabels.add(new TokenLabel(token, label, -1.0));
				logger.fine(a + ": " + label);
			}
			NerEntry ne = new NerEntry(getOriginalToken(token, sentence), token, bestLabel, nLabels);
			nerEntries.add(ne);
		}
		return nerEntries;
	}
	/**
	 * Apply a transducer to an input sequence to produce the k highest-scoring
	 * output sequences.
	 *
	 * @param model the <code>Transducer</code>
	 * @param input the input sequence
	 * @param k the number of answers to return
	 * @return array of the k highest-scoring output sequences
	 */
	public Sequence<?>[] apply(Transducer model, Sequence<?> input, int k){
		Sequence<?>[] answers;
		if (k == 1){
			answers = new Sequence[1];
			answers[0] = model.transduce(input);
		}else{
			MaxLatticeDefault lattice =	new MaxLatticeDefault(model, input, null, cacheSize);
			answers = lattice.bestOutputSequences(k).toArray(new Sequence[0]);
		}
		return answers;
	}
	
	@Override
	public List<NerEntry> getEntities(String sentence, boolean fuseSame, boolean removeDefaultLabel) {
		//Normalize and get tokens
		List<String> tokens = tokenizer.getTokens(sentence);
		
		//Create instance - TODO: there has got to be an easier way ...
		String data = String.join("\n", tokens);		//this breaks it: System.getProperty("line.separator")
		Instance carrier = new Instance(data, null, "linegroup0", null);
		
		//Get pipe and add instance
		Pipe p = this.crfModel.getInputPipe();
		p.setTargetProcessing(false);
		InstanceList testData = new InstanceList(p);
		testData.addThruPipe(carrier);
		
		Instance inst = testData.get(0);
		Sequence<?> input = (Sequence<?>) inst.getData();
		Sequence<?>[] outputs = apply(this.crfModel, input, nBest);
		
		List<NerEntry> nerEntries = new ArrayList<>();
		NerEntry ne = null;
		String lastLabel = "";
		
		//filter special labels
		String commonlabelRegex = RealLifeChatTokenizer.getCommonLabelRegEx(tokenizer, removeDefaultLabel);
		
		for (int j = 0; j < input.size(); j++){
			String token = tokens.get(j);
			String bestLabel = "";
			List<TokenLabel> nLabels = new ArrayList<>();
			for (int a = 0; a < outputs.length; a++){
				String label = outputs[a].get(j).toString();
				if (a == 0){
					bestLabel = label;
				}
				nLabels.add(new TokenLabel(token, label, -1.0));
				logger.fine(a + ": " + label);
			}

			//add to previous entry?
			if (fuseSame && ne != null && bestLabel.equals(lastLabel)){
				ne.addToken(token);
				ne.addOriginalToken(getOriginalToken(token, sentence));
				//NOTE: if you fuse two labels and they had different "alternatives" this might make certainties unreliable 
				ne.addAllLabels(nLabels);
			
			//make new entry
			}else{
				//skip all labels that are common
				if (!bestLabel.matches(commonlabelRegex)){
					lastLabel = bestLabel;
					ne = new NerEntry(getOriginalToken(token, sentence), token, bestLabel, nLabels);
					nerEntries.add(ne);
				}else{
					ne = null;
				}
			}
		}
		return nerEntries;
	}
	
	//This is a kind of "heavy" way to get back the original token before it was converted to lower-case ... :see_no_evil:
	private String getOriginalToken(String normToken, String orgSentence){
		Matcher matcher = Pattern.compile("(?i)" + normToken).matcher(orgSentence);
		if (matcher.find()){
			return matcher.group();
		}else{
			return normToken;
		}
	}

}
