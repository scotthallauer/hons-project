package za.redbridge.experiment.MultiObjective;

import org.encog.ml.CalculateScore;
import org.encog.ml.ea.codec.GeneticCODEC;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.score.AdjustScore;
import org.encog.ml.ea.score.parallel.ParallelScore;
import org.encog.ml.ea.score.parallel.ParallelScoreTask;
import org.encog.ml.ea.species.Species;
import org.encog.ml.genetic.GeneticError;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MultiObjectiveParallelScore extends ParallelScore
{

    /**
     * The actual number of threads.
     */
    private int actualThreads;

    public MultiObjectiveParallelScore(Population thePopulation, GeneticCODEC theCODEC,
                         List<AdjustScore> theAdjusters, CalculateScore theScoreFunction,
                         int theThreadCount) {
        super(thePopulation,theCODEC,theAdjusters,theScoreFunction,theThreadCount);
        this.actualThreads = 0;
    }
    /**
     * Calculate the scores.
     */
    @Override
    public void process() {
        // determine thread usage
        if (this.getScoreFunction().requireSingleThreaded()) {
            this.actualThreads = 1;
        } else if (super.getThreadCount()== 0) {
            this.actualThreads = Runtime.getRuntime().availableProcessors();
        } else {
            this.actualThreads = getThreadCount();
        }

        // start up
        ExecutorService taskExecutor = null;

        if (getThreadCount() == 1) {
            taskExecutor = Executors.newSingleThreadScheduledExecutor();
        } else {
            taskExecutor = Executors.newFixedThreadPool(this.actualThreads);
        }

        for (Species species : getPopulation().getSpecies()) {
            for (Genome genome : species.getMembers()) {
                taskExecutor.execute(new MultiObjectiveParallelScoreTask(genome, this));
            }
        }

        taskExecutor.shutdown();
        try {
            taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            throw new GeneticError(e);
        }
    }
}
