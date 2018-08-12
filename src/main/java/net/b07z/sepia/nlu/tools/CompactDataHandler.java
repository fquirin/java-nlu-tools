package net.b07z.sepia.nlu.tools;

import java.util.Collection;
import java.util.List;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;

public interface CompactDataHandler {
	
	/**
	 * Import data from compact format for NER and export to a list in required format that can e.g. be written to a train-file.
	 * Uses the tokenizer to get tokens from sentence and adds beginning/end-of-sentence tokens automatically (if set).
	 * @param compactData - collection of imported {@link CompactDataEntry} objects
	 * @param tokenizer - for Mallet it's recommended to NOT use BOS and EOS tags
	 * @param failOnMismatch - fail or skip on token-label mismatch?
	 * @param filterLabel - if given (not null or empty), only this label will be tagged in the data
	 * @return
	 */
	public List<String> importTrainDataNer(Collection<CompactDataEntry> compactData, Tokenizer tokenizer, boolean failOnMismatch, String filterLabel);
	
	/**
	 * Import data from compact format for intent recognition and export to a list in required format that can e.g. be written to a train-file.
	 * Uses the tokenizer to normalize the sentence
	 * @param compactData - collection of imported {@link CompactDataEntry} objects
	 * @param tokenizer - for Mallet it's recommended to NOT use BOS and EOS tags
	 * @param filterIntent - if given (not null or empty), only sentences for this intent will be imported
	 * @return
	 */
	public List<String> importTrainDataIntent(Collection<CompactDataEntry> compactData, Tokenizer tokenizer, String filterIntent);

}
