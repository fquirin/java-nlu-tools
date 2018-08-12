package net.b07z.sepia.nlu.classifiers;

import java.util.Comparator;
import java.util.Set;
import java.util.TreeSet;

/**
 * Holds the resulting labels of a token for each model, sorted by certainty.
 */
public class TokenModelLabels {
	String token;
	Set<TokenLabel> tokenLabels;
	
	public TokenModelLabels(String token){
		this.token = token;
		//Order descending by certainty, if certainty is equal and label is NOT equal keep order
		Comparator<TokenLabel> comp = (TokenLabel l1, TokenLabel l2) -> { 
			int dc = Double.compare(l2.getCertainty(), l1.getCertainty());
			if (dc == 0){
				int sc = l1.getLabel().compareTo(l2.getLabel());
				if (sc == 0) return 0;	//if a label has same name and certainty it is removed from set
				else return 1;			//keep input order in this case
			}else{
				return dc;
			}
		};
		this.tokenLabels = new TreeSet<>(comp);
	}
	
	public void addLabel(TokenLabel tl){
		tokenLabels.add(tl);
	}
	public void addLabel(String label, double certainty){
		tokenLabels.add(new TokenLabel(this.token, label, certainty));
	}
	
	public String getBestLabel(){
		return tokenLabels.iterator().next().getLabel();
	}
	public Set<TokenLabel> getAllLabels(){
		return tokenLabels;
	}
	
	public double getBestCertainty(){
		return tokenLabels.iterator().next().getCertainty();
	}
}