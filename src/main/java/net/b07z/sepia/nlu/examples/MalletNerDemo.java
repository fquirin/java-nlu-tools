package net.b07z.sepia.nlu.examples;

import java.util.Collection;
import java.util.List;
import net.b07z.sepia.nlu.classifiers.MalletNerClassifier;
import net.b07z.sepia.nlu.classifiers.NerClassifier;
import net.b07z.sepia.nlu.classifiers.NerEntry;
import net.b07z.sepia.nlu.tokenizers.RealLifeChatTokenizer;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;
import net.b07z.sepia.nlu.tools.CompactDataEntry;
import net.b07z.sepia.nlu.tools.CompactDataHandler;
import net.b07z.sepia.nlu.tools.CustomDataHandler;
import net.b07z.sepia.nlu.tools.MalletDataHandler;
import net.b07z.sepia.nlu.trainers.MalletNerTrainer;
import net.b07z.sepia.nlu.trainers.NerTrainer;

/**
 * Conditional-Random-Field (CRF) NER classifier with MALLET. 
 * 
 * @author Florian Quirin
 *
 */
public class MalletNerDemo {

	public static void main(String[] args) throws Exception {
		long tic = System.currentTimeMillis();
		System.setProperty("java.util.logging.config.file", "resources/logging.properties");
		
		String nerCompactTrainDataFile = "data/nerCompactTrain.txt";
		String nerCompactTestDataFile = "data/nerCompactTest.txt";
		String nerTrainerPropertiesFile = null; //"data/malletner.properties";
		String nerModelFileBase = "data/mallet_model_ner";
		String nerTrainFileBase = "data/mallet_train_ner";
		
		String languageCode = "en";
		
		//Create training data from compact custom format
		Collection<CompactDataEntry> trainData = CustomDataHandler.importCompactData(nerCompactTrainDataFile);
		Collection<CompactDataEntry> testData = CustomDataHandler.importCompactData(nerCompactTestDataFile);
		
		//Convert compact format to MALLET format and write file
		Tokenizer tokenizer = new RealLifeChatTokenizer("", "", "SEP"); 		//do we want ESO and BOS for Mallet???
		//Tokenizer tokenizer = new RealLifeChatTokenizer();
		CompactDataHandler cdh = new MalletDataHandler();
		List<String> trainDataLines = cdh.importTrainDataNer(trainData, tokenizer, false, null);
		String trainFile = nerTrainFileBase + "_" + languageCode;
		CustomDataHandler.writeTrainData(trainFile, trainDataLines);
				
		//Train - NOTE: we used the tokenizer before to generate tokens and now we assume that we can simply split at whitespace
		NerTrainer trainer = new MalletNerTrainer(nerTrainerPropertiesFile, nerTrainFileBase, nerModelFileBase, languageCode);
		trainer.train();
		
		//Test
		NerClassifier ner = new MalletNerClassifier(nerModelFileBase, tokenizer, languageCode);
		int good = 0;
		int bad = 0;
		for (CompactDataEntry cde : testData){
			String sentence = cde.getSentence();
			List<NerEntry> nerRes = ner.getEntities(sentence, false, false);
			String labelsAsString = NerEntry.getLabelStringFromEntryList(nerRes);
			String labelsExpected = cde.getLabels();
			System.out.println(sentence);
			System.out.println(NerEntry.getTokenStringFromEntryList(nerRes));
			if (labelsExpected.equals(labelsAsString)){
				System.out.println(labelsExpected);
				System.out.println(labelsAsString);
				good++;
			}else{
				System.out.println(labelsExpected);
				System.out.println(labelsAsString + " --- ERROR");
				bad++;
			}
			System.out.println();
		}
		System.out.println("Good: " + good + ", bad: " + bad + ", prec.: " + ((double)good/(good+bad)));
		System.out.println("Took: " + (System.currentTimeMillis() - tic) + "ms");
	}

}
