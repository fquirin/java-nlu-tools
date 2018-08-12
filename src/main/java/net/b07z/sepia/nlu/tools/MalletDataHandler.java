package net.b07z.sepia.nlu.tools;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;

/**
 * Writes data in different formats to a file that can be used to train MALLET NER.
 * 
 * @author Florian Quirin
 *
 */
public class MalletDataHandler implements CompactDataHandler {
	
	@Override
	public List<String> importTrainDataNer(Collection<CompactDataEntry> compactData, Tokenizer tokenizer, boolean failOnMismatch, String filterLabel){
		List<String> trainDataLines = new ArrayList<>();
		String defaultLabel = tokenizer.getDefaultLabel();
		int n = 0;
		for (CompactDataEntry cde : compactData){
			n++;
			String sentence = cde.getSentence();
			String labelString = cde.getLabels();
			if (labelString == null){
				if (failOnMismatch){
					throw new RuntimeException("Skipped sentence - Missing labels in line " + n + ": " + sentence);
				}else{
					System.err.println("Skipped sentence - Missing labels in line " + n + ": " + sentence);
				}
				continue;
			}
			labelString = tokenizer.getBeginningOfSentenceLabel() + labelString + tokenizer.getEndOfSentenceLabel();
			List<String> tokens = tokenizer.getTokens(sentence);
			String[] labels = labelString.trim().split("\\s+");
			boolean gotError = false;
			if (tokens.size() != labels.length){
				if (failOnMismatch){
					throw new RuntimeException("Tokens and labels size does not match in line " + n + ": " + sentence);
				}else{
					System.err.println("Skipped sentence - Tokens and labels size does not match in line " + n + ": " + sentence);
				}
				gotError = true;
			}
			if (!gotError){
				for (int i=0; i<tokens.size(); i++){
					if (filterLabel != null && !filterLabel.isEmpty() && !labels[i].equals(filterLabel)){
						trainDataLines.add(tokens.get(i) + " " + defaultLabel);
					}else{
						trainDataLines.add(tokens.get(i) + " " + labels[i]);
					}
				}
				trainDataLines.add(" "); 		//Mallet uses empty line to separate sentences by default
			}
		}
		return trainDataLines;
	}

	@Override
	public List<String> importTrainDataIntent(Collection<CompactDataEntry> compactData, Tokenizer tokenizer,
			String filterIntent) {
		throw new RuntimeException("Not yet implemented");
	}
}
