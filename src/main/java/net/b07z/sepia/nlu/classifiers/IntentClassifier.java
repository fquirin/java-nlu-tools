package net.b07z.sepia.nlu.classifiers;

import java.util.Collection;

/**
 * Classifier interface to analyze text and extract intents.
 * 
 * @author Florian Quirin
 *
 */
public interface IntentClassifier {
	
	/**
	 * Return list with intent results for a sentence.
	 */
	public Collection<IntentEntry> analyzeSentence(String sentence);
}
