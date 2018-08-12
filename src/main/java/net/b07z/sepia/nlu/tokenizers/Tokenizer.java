package net.b07z.sepia.nlu.tokenizers;

import java.util.List;

/**
 * Interface for classes that create tokens (words etc.) of sentences and sentences of texts.
 * 
 * @author Florian Quirin
 *
 */
public interface Tokenizer {
	
	/**
	 * Convert a sentence to a list of tokens.
	 * @param sentence - raw text input
	 * @return normalized tokens
	 */
	public List<String> getTokens(String sentence);
	
	/**
	 * Convert sentence to tokens and return joined tokens.
	 * @param sentence - raw text input
	 * @return
	 */
	public String normalizeSentence(String sentence);
	
	/**
	 * Get label used to mark beginning of sentence.
	 */
	public String getBeginningOfSentenceLabel();
	
	/**
	 * Get label used to mark end of sentence.
	 */
	public String getEndOfSentenceLabel();
	
	/**
	 * Get label used to mark a seperator between tokens.
	 */
	public String getSeparatorLabel();
	
	/**
	 * Get the default label for "unlabeled" tokens. NOTE: it has to be "O" right now. 
	 */
	public String getDefaultLabel();

}
