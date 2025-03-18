package de.unimarburg.generateddatasetdetector.matching.similarity.label;

import de.unimarburg.generateddatasetdetector.similarity.string.JaroWinkler;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class JaroWinklerMatcher extends LabelSimilarityMatcher {
    public JaroWinklerMatcher() {
        super(new JaroWinkler());
    }
}
