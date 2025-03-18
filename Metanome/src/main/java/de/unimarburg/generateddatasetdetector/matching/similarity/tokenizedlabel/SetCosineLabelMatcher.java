package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedlabel;

import lombok.EqualsAndHashCode;
import de.unimarburg.generateddatasetdetector.similarity.set.SetCosine;


@EqualsAndHashCode(callSuper = true)
public class SetCosineLabelMatcher extends TokenizedLabelSimilarityMatcher {
    public SetCosineLabelMatcher() {
        super(new SetCosine<>());
    }
}
