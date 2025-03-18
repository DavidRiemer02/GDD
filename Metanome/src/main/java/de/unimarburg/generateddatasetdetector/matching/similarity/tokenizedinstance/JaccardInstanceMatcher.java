package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedinstance;

import de.unimarburg.generateddatasetdetector.similarity.set.Jaccard;

import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class JaccardInstanceMatcher extends TokenizedInstanceSimilarityMatcher {
    public JaccardInstanceMatcher() {
        super(new Jaccard<>());
    }
}
