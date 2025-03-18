package de.unimarburg.generateddatasetdetector.evaluation.metric;

import de.unimarburg.generateddatasetdetector.evaluation.performance.Performance;

public abstract class Metric {
    public abstract float run(int[] groundTruthVector, float[] simVector);

    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
