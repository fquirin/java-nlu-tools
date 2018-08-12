package net.b07z.sepia.nlu.examples;

import java.util.Collection;
import java.util.List;
import net.b07z.sepia.nlu.classifiers.NerClassifier;
import net.b07z.sepia.nlu.classifiers.NerEntry;
import net.b07z.sepia.nlu.classifiers.OpenNlpNerClassifier;
import net.b07z.sepia.nlu.tokenizers.RealLifeChatTokenizer;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;
import net.b07z.sepia.nlu.tools.CompactDataEntry;
import net.b07z.sepia.nlu.tools.CompactDataHandler;
import net.b07z.sepia.nlu.tools.CustomDataHandler;
import net.b07z.sepia.nlu.tools.OpenNlpDataHandler;
import net.b07z.sepia.nlu.trainers.NerTrainer;
import net.b07z.sepia.nlu.trainers.OpenNlpNerTrainer;

/**
 * Maximum-Entropy (ME) NER classifier with OpenNLP. 
 * 
 * @author Florian Quirin
 *
 */
public class OpenNlpNerDemo {

	public static void main(String[] args) throws Exception {
		long tic = System.currentTimeMillis();
		System.setProperty("java.util.logging.config.file", "resources/logging.properties");
		
		String nerCompactTrainDataFile = "data/nerCompactTrain.txt";
		String nerCompactTestDataFile = "data/nerCompactTest.txt";
		String nerTrainerPropertiesFile = null;
		String nerModelFileBase = "data/opennlp_model_ner";
		String nerTrainFileBase = "data/opennlp_train_ner";
		
		//Label to search
		String[] labels = new String[]{"LOC_END", "LOC_START"};
		String languageCode = "en";
		
		//Create training data from compact custom format
		Collection<CompactDataEntry> trainData = CustomDataHandler.importCompactData(nerCompactTrainDataFile);
		Collection<CompactDataEntry> testData = CustomDataHandler.importCompactData(nerCompactTestDataFile);
		
		//Tokenizer
		Tokenizer tokenizer = new RealLifeChatTokenizer("", "", "SEP"); 		//do we want ESO and BOS here???
		//Tokenizer tokenizer = new RealLifeChatTokenizer();
		
		//Get train data from compact format
		CompactDataHandler cdh = new OpenNlpDataHandler();
		for (String label : labels){
			String nerTrainFile = nerTrainFileBase + "_" + label + "_" + languageCode;
			//Convert compact format to OpenNLP format and write file
			List<String> trainDataLines = cdh.importTrainDataNer(trainData, tokenizer, false, label);
			CustomDataHandler.writeTrainData(nerTrainFile, trainDataLines);
		}
		
		//Start training
		NerTrainer trainer = new OpenNlpNerTrainer(nerTrainerPropertiesFile, 
				nerTrainFileBase, nerModelFileBase, labels, languageCode);
		trainer.train();
						
		//Test
		int good = 0;
		int bad = 0;
		NerClassifier ner = new OpenNlpNerClassifier(nerModelFileBase, tokenizer, labels, languageCode);
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
