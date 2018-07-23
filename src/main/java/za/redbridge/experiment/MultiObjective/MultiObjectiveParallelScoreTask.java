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
            try
            {
                score = this.scoreFunction.calculateScore(phenotype);
                score2 = this.scoreFunction.calculateScore2(phenotype);

            }
            catch (EARuntimeError e)
            {
                score = Double.NaN;
                score2 = Double.NaN;
            }

            if (genome.getClass().equals(MultiObjectiveHyperNEATGenome.class))
            {
                ((MultiObjectiveHyperNEATGenome) genome).addScore(score);
                ((MultiObjectiveHyperNEATGenome) genome).addScore(score2);
            }
            else if(genome.getClass().equals(MultiObjectiveNEATMGenome.class))
            {

                ((MultiObjectiveNEATMGenome) genome).addScore(score);
                ((MultiObjectiveNEATMGenome) genome).addScore(score2);
            }
            else
            {
                throw new ClassCastException("Genome isn't of any multi-objective type.");
            }


            genome.setScore(score);
            genome.setAdjustedScore(score);
            BasicEA.calculateScoreAdjustment(genome, adjusters);
        } else {

        }
    }
}
