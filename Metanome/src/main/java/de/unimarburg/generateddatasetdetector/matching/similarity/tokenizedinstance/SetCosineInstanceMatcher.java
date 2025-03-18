package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedinstance;

import lombok.EqualsAndHashCode;
import de.unimarburg.generateddatasetdetector.similarity.set.SetCosine;

@EqualsAndHashCode(callSuper = true)
public class SetCosineInstanceMatcher extends TokenizedInstanceSimilarityMatcher {
    public SetCosineInstanceMatcher() {
        super(new SetCosine<>());
    }
}
