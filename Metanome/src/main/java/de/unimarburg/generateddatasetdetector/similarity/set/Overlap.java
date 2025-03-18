package de.unimarburg.generateddatasetdetector.similarity.set;

import de.unimarburg.generateddatasetdetector.similarity.SimilarityMeasure;
import de.unimarburg.generateddatasetdetector.utils.SetUtils;

import java.util.Set;

public class Overlap<T> implements SimilarityMeasure<Set<T>> {
    @Override
    public float compare(Set<T> source, Set<T> target) {
        float score = 0;
        int minSize = Math.min(source.size(), target.size());
        Set<T> intersection = SetUtils.intersection(source, target);
        if (minSize > 0) {
            score = (float) intersection.size() / minSize;
        }
        return score;
    }
}
