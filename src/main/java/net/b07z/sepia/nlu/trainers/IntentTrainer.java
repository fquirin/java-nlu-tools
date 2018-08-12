package net.b07z.sepia.nlu.trainers;

/**
 * Interface for training intent classifier models.
 * 
 * @author Florian Quirin
 *
 */
public interface IntentTrainer {
	
	/**
	 * Train model with info given in constructor and store it.
	 */
	public void train();
}
