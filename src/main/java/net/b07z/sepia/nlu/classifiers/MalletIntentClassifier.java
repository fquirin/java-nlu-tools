package net.b07z.sepia.nlu.classifiers;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Collection;
import java.util.List;
import java.util.logging.Logger;
import cc.mallet.classify.Classifier;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;
import cc.mallet.util.MalletLogger;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;

/**
 * MALLET implementation of ME intent classifier.
 * 
 * @author Florian Quirin
 *
 */
public class MalletIntentClassifier implements IntentClassifier{
	
	private static Logger logger = MalletLogger.getLogger(MalletIntentClassifier.class.getName());
	
	private Classifier classifierModel;
	private Tokenizer tokenizer;
	private String languageCode;
	
	private int nBest = 3; 				//How many answers to output
	
	/**
	 * Create intent classifier with model and tokenizer.
	 * @param modelFileBase
	 * @param tokenizer
	 * @param languageCode
	 */
	public MalletIntentClassifier(String modelFileBase, Tokenizer tokenizer, String languageCode) throws Exception {
		this.languageCode = languageCode;
		//load model
		String modelFile = modelFileBase + "_" + this.languageCode;
		try (ObjectInputStream s = new ObjectInputStream(new FileInputStream(modelFile))){
			this.classifierModel = (Classifier) s.readObject();
		}catch(Exception e){
			throw e;
		}
		this.tokenizer = tokenizer;
	}

	@Override
	public Collection<IntentEntry> analyzeSentence(String sentence) {
		//Normalize and get tokens
		List<String> tokens = tokenizer.getTokens(sentence);
		
		//Create instance
		String data = String.join(" ", tokens);	//if you want to use System.getProperty("line.separator") check SentenceToFeatureVectorSequencePipe
		Instance carrier = new Instance(data, "OTHER", "userinput", null); 		//TODO: the labels is arbitrary but has to have existing value
		
		//Get pipe and add instance
		Pipe p = this.classifierModel.getInstancePipe();
		p.setTargetProcessing(false);
		InstanceList instances = new InstanceList(p);
		instances.addThruPipe(carrier);

		//Collect
		Collection<IntentEntry> intents = IntentEntry.makeIntentCollection();
		for (Instance insta : instances){
	        Labeling labeling = this.classifierModel.classify(insta).getLabeling();
	        // print the labels with their weights in descending order (ie best first)                     
	        for (int rank = 0; rank < Math.min(labeling.numLocations(), this.nBest); rank++){
	        	intents.add(new IntentEntry(data, labeling.getLabelAtRank(rank).toString(), labeling.getValueAtRank(rank)));
	        	logger.fine(labeling.getLabelAtRank(rank) + ":" + labeling.getValueAtRank(rank) + " ");
	        }
		}
		return intents;
	}
}
