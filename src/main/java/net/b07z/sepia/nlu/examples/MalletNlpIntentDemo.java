package net.b07z.sepia.nlu.examples;

import java.util.Collection;
import java.util.List;

import net.b07z.sepia.nlu.classifiers.IntentClassifier;
import net.b07z.sepia.nlu.classifiers.IntentEntry;
import net.b07z.sepia.nlu.classifiers.MalletIntentClassifier;
import net.b07z.sepia.nlu.tokenizers.RealLifeChatTokenizer;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;
import net.b07z.sepia.nlu.tools.CompactDataEntry;
import net.b07z.sepia.nlu.tools.CompactDataHandler;
import net.b07z.sepia.nlu.tools.CustomDataHandler;
import net.b07z.sepia.nlu.tools.MalletDataHandler;
import net.b07z.sepia.nlu.trainers.IntentTrainer;
import net.b07z.sepia.nlu.trainers.MalletIntentTrainer;

/**
 * Maximum-Entropy (ME) intent classifier with Mallet. 
 * 
 * @author Florian Quirin
 *
 */
public class MalletNlpIntentDemo {

	public static void main(String[] args) throws Exception {
		long tic = System.currentTimeMillis();
		System.setProperty("java.util.logging.config.file", "resources/logging.properties");
		
		String compactTrainDataFile = "data/intentCompactTrain.txt";
		String compactTestDataFile = "data/intentCompactTest.txt";

		String modelFileBase = "data/mallet_model_intents";
		String trainFileBase = "data/mallet_train_intents";
		
		String languageCode = "en";
		
		//Create training data from compact custom format
		Collection<CompactDataEntry> trainData = CustomDataHandler.importCompactData(compactTrainDataFile);
		Collection<CompactDataEntry> testData = CustomDataHandler.importCompactData(compactTestDataFile);
		
		//Tokenizer
		Tokenizer tokenizer = new RealLifeChatTokenizer("", "", ""); 		//do we want ESO and BOS here???
		//Tokenizer tokenizer = new RealLifeChatTokenizer();
		
		//Get train data from compact format
		CompactDataHandler cdh = new MalletDataHandler();
		String trainFile = trainFileBase + "_" + languageCode;
		//Convert compact format to OpenNLP format and write file
		List<String> trainDataLines = cdh.importTrainDataIntent(trainData, tokenizer, null);
		CustomDataHandler.writeTrainData(trainFile, trainDataLines);
		
		//Start training
		IntentTrainer trainer = new MalletIntentTrainer(null, trainFileBase, modelFileBase, languageCode);
		trainer.train();
						
		//Test
		int good = 0;
		int bad = 0;
		double cThresh = 0.33;
		double certaintySumBest = 0.0;
		IntentClassifier ic = new MalletIntentClassifier(modelFileBase, tokenizer, languageCode);
		for (CompactDataEntry cde : testData){
			String sentence = cde.getSentence();
			Collection<IntentEntry> intentRes = ic.analyzeSentence(sentence);
			if (!intentRes.isEmpty()){
				IntentEntry bestRes = IntentEntry.getBestIntent(intentRes);
				String intentExpected = cde.getIntent();
				String bestIntent = bestRes.getIntent();
				double bestCertainty = bestRes.getCertainty();
				System.out.println(sentence);
				System.out.println(bestRes.getSentence());
				System.out.println("Expected: " + intentExpected);
				if (intentExpected.equals(bestIntent) && (bestCertainty > cThresh)){
					System.out.println("Classif.: " + bestRes.toString());
					good++;
					certaintySumBest += bestCertainty;
				}else{
					System.out.println("Classif.: " + bestRes.toString() + " --- FALSE");
					bad++;
				}
				int N = 3;
				for (IntentEntry ie : intentRes){
					System.out.println(ie.toStringComplex());
					if (--N <= 0) break;
				}
				System.out.println();
			}else{
				System.err.println("Nothing found for: " + sentence);
				bad++;
			}
		}
		System.out.println("Good: " + good + ", bad: " + bad + ", prec.: " + ((double)good/(good+bad)));
		System.out.println("Certainty average (good): " + certaintySumBest/good);
		System.out.println("Took: " + (System.currentTimeMillis() - tic) + "ms");
	}

}
