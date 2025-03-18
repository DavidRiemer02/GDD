package de.unimarburg.generateddatasetdetector.similarity.string;

import de.unimarburg.generateddatasetdetector.similarity.SimilarityMeasure;
import org.apache.commons.text.similarity.JaroWinklerDistance;

public class JaroWinkler implements SimilarityMeasure<String> {
    @Override
    public float compare(String s1, String s2) {
        return new JaroWinklerDistance().apply(s1, s2).floatValue();
    }
}
