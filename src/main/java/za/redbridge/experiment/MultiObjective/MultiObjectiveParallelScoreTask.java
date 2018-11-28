package za.redbridge.experiment.MultiObjective;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.ea.exception.EARuntimeError;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.score.AdjustScore;
import org.encog.ml.ea.score.parallel.ParallelScore;
import org.encog.ml.ea.score.parallel.ParallelScoreTask;
import org.encog.ml.ea.train.basic.BasicEA;
import za.redbridge.experiment.ScoreCalculator;

import java.util.List;

public class MultiObjectiveParallelScoreTask implements Runnable
{/**
 * The genome to calculate the score for.
 */
private final Genome genome;

    /**
     * The score function.
     */
    private final ScoreCalculator scoreFunction;

    /**
     * The score adjusters.
     */
    private final List<AdjustScore> adjusters;

    /**
     * The owners.
     */
    private final ParallelScore owner;

    /**
     * Construct the parallel task.
     * @param genome The genome.
     * @param theOwner The owner.
     */
    public MultiObjectiveParallelScoreTask(Genome genome, ParallelScore theOwner) {
        super();
        this.owner = theOwner;
        this.genome = genome;
        this.scoreFunction = (ScoreCalculator) theOwner.getScoreFunction();
        this.adjusters = theOwner.getAdjusters();
    }

    /**
     * Perform the task.
     */
    @Override
    public void run()
    {
        MLMethod phenotype = this.owner.getCodec().decode(this.genome);
        if (phenotype != null)
        {
            double score;
            double score2;
            double score3;
            try
            {
                score = this.scoreFunction.calculateScore(phenotype);
                score2 = this.scoreFunction.calculateScore2(phenotype);
                score3 = this.scoreFunction.calculateScore3(phenotype);
            }
            catch (EARuntimeError e)
            {
                score = Double.NaN;
                score2 = Double.NaN;
                score3 = Double.NaN;
            }

            ((MultiObjectiveGenome) genome).setScore(0,score);
            ((MultiObjectiveGenome) genome).setScore(1, score2);
            ((MultiObjectiveGenome) genome).setScore(2, score3);
            genome.setScore(score);
            genome.setAdjustedScore(score);
            BasicEA.calculateScoreAdjustment(genome, adjusters);
        }
    }
}
