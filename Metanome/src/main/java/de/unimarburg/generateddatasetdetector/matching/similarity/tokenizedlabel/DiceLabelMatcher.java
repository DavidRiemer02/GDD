package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedlabel;

import de.unimarburg.generateddatasetdetector.similarity.set.Dice;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class DiceLabelMatcher extends TokenizedLabelSimilarityMatcher {
    public DiceLabelMatcher() {
        super(new Dice<>());
    }
}
