package net.b07z.sepia.nlu.classifiers;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestTokenLabels {

	@Test
	public void test() {
		TokenModelLabels ml = new TokenModelLabels("Valhalla");
		ml.addLabel("LOC_START", 0.5);
		ml.addLabel("LOC_END", 0.85);
		ml.addLabel("POSITION", 0.45);
		ml.addLabel("PLACE", 0.45);
		
		assertTrue(ml.getBestLabel().equals("LOC_END"));
		assertTrue(ml.getBestCertainty() == 0.85);
		StringBuffer sb = new StringBuffer();
		ml.tokenLabels.forEach((tl) -> {
			String l = tl.toStringComplex();
			sb.append(l).append(", ");
		});
		String res = sb.toString();
		//System.out.println(res);
		assertTrue(res.equals("LOC_END (0.85), LOC_START (0.5), POSITION (0.45), PLACE (0.45), "));
	}

}
