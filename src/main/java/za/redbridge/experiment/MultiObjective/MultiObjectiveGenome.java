package za.redbridge.experiment.MultiObjective;

import org.encog.ml.ea.genome.BasicGenome;
import org.encog.ml.ea.genome.Genome;

import java.util.ArrayList;

public interface MultiObjectiveGenome extends Genome
{
    public abstract void setScore(int objective, Double score);

    public abstract ArrayList<Double> getScoreVector();

    public abstract void setDistance(Double distance);

    public abstract void setRank(int rank);

    public abstract double getDistance();

    public abstract int getRank();

}
