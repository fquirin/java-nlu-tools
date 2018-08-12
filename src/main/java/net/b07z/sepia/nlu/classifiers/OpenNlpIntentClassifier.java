package net.b07z.sepia.nlu.classifiers;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;

/**
 * OpenNLP implementation of intent classifier interface.
 * 
 * @author Florian Quirin
 *
 */
public class OpenNlpIntentClassifier implements IntentClassifier {
	
	private Tokenizer tokenizer;
	private DoccatModel model;
	private String languageCode;
	
	/**
	 * Create classifier by loading model from file.
	 * @param modelFileBase
	 * @param tokenizer
	 * @param languageCode
	 * @throws Exception
	 */
	public OpenNlpIntentClassifier(String modelFileBase, Tokenizer tokenizer, String languageCode) throws Exception {
		this.languageCode = languageCode;
		String modelFile = modelFileBase + "_" + this.languageCode;
		try (InputStream modelIn = new FileInputStream(modelFile)){
			this.model = new DoccatModel(modelIn);
		}
		this.tokenizer = tokenizer;
	}

	@Override
	public Collection<IntentEntry> analyzeSentence(String sentence) {
		if (this.model == null){
			throw new RuntimeException("Model was null!");
		}
		//Normalize and get tokens
		List<String> tokens = tokenizer.getTokens(sentence);
		String normalizedSentence = String.join(" ", tokens); 		//TODO: we use space to join but this does not have to be right ...
		
		//Classify
		DocumentCategorizerME classifier = new DocumentCategorizerME(this.model);
		//double[] outcomes = classifier.categorize(tokens.toArray(new String[tokens.size()]));
		//String intent = classifier.getBestCategory(outcomes);
		//System.out.println(classifier.getAllResults(outcomes));
		
		//Collect
		Collection<IntentEntry> intents = IntentEntry.makeIntentCollection();
		Map<String, Double> ssm = classifier.scoreMap(tokens.toArray(new String[tokens.size()]));
		for (Entry<String, Double> es : ssm.entrySet()){
			intents.add(new IntentEntry(normalizedSentence, es.getKey(), es.getValue()));
		}
		
		return intents;
	}	
}
