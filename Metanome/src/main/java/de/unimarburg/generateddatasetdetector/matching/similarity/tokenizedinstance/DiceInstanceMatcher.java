package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedinstance;

import de.unimarburg.generateddatasetdetector.similarity.set.Dice;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class DiceInstanceMatcher extends TokenizedInstanceSimilarityMatcher {
    public DiceInstanceMatcher() {
        super(new Dice<>());
    }
}
