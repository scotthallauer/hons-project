package za.redbridge.experiment;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SynchronizedDescriptiveStatistics;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.ea.genome.Genome;
import org.encog.neural.neat.NEATNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

import sim.display.Console;
import za.redbridge.experiment.HyperNEATM.HyperNEATMCODEC;
import za.redbridge.experiment.NEATM.NEATMNetwork;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.simulator.Simulation;
import za.redbridge.simulator.SimulationGUI;
import za.redbridge.simulator.config.SimConfig;
import za.redbridge.simulator.factories.HomogeneousRobotFactory;
import za.redbridge.simulator.factories.RobotFactory;
import za.redbridge.simulator.phenotype.Phenotype;

/**
 * Test runner for the simulation.
 * <p>
 * Created by jamie on 2014/09/09.
 */
public class ScoreCalculator implements CalculateScore
{

    private static final Logger log = LoggerFactory.getLogger(ScoreCalculator.class);

    private final SimConfig simConfig;
    private final int trialsPerIndividual;
    private final SensorMorphology sensorMorphology;

    private final DescriptiveStatistics performanceStats = new SynchronizedDescriptiveStatistics();
    private final DescriptiveStatistics scoreStats = new SynchronizedDescriptiveStatistics();
    private final DescriptiveStatistics sensorStats;
    private boolean hyperNEATM;

    public ScoreCalculator(SimConfig simConfig, int trialsPerIndividual,
                           SensorMorphology sensorMorphology, boolean hyperNEAT )
    {
        this.simConfig = simConfig;
        this.trialsPerIndividual = trialsPerIndividual;
        this.sensorMorphology = sensorMorphology;
        this.hyperNEATM = hyperNEATM;

        // If fixed morphology then don't record sensor stats
        this.sensorStats = isEvolvingMorphology() ? new SynchronizedDescriptiveStatistics() : null;
    }

    @Override
    public double calculateScore(MLMethod method)
    {
        long start = System.nanoTime();

        NEATNetwork network = (NEATNetwork) method;
        RobotFactory robotFactory = new HomogeneousRobotFactory(getPhenotypeForNetwork(network),
                simConfig.getRobotMass(), simConfig.getRobotRadius(), simConfig.getRobotColour(),
                simConfig.getObjectsRobots());

        // Create the simulation and run it
        Simulation simulation = new Simulation(simConfig, robotFactory);
        simulation.setStopOnceCollected(true);
        double fitness = 0;
        for (int i = 0; i < trialsPerIndividual; i++)
        {
            simulation.run();
            fitness += simulation.getFitness().getTeamFitness();
            fitness += 20 * (1.0 - simulation.getProgressFraction()); // Time bonus --> progress fraction is ratio of timesteps that were needed to gather all resources with respect to
                                                                      // total possible timesteps for each individual's trial. So if an individual doesn't gather all resources, progress fraction
                                                                      // will just land up being 1, and thus the time bonus will be 20*(1-1) = 0.
        }

        // Get the fitness and update the total score
        double score = fitness / trialsPerIndividual;
        scoreStats.addValue(score);

        if (isEvolvingMorphology() || hyperNEATM)
        {
            sensorStats.addValue(network.getInputCount());
        }

        log.debug("Score calculation completed: " + score);

        long duration = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
        performanceStats.addValue(duration);

        return score;
    }

    public double calculateScore2(MLMethod method)  // second fitness function score (multi-objective)
    {
        NEATNetwork network = (NEATNetwork) method;
        System.out.println(network.getInputCount());

        double score = 100 - (( (double) network.getInputCount() / 11) * 100);
        return score;

    }

    public void demo(MLMethod method)
    {
        // Create the robot and resource factories
        NEATNetwork network = (NEATNetwork) method;
        RobotFactory robotFactory = new HomogeneousRobotFactory(getPhenotypeForNetwork(network),
                simConfig.getRobotMass(), simConfig.getRobotRadius(), simConfig.getRobotColour(),
                simConfig.getObjectsRobots());

        // Create the simulation and run it
        Simulation simulation = new Simulation(simConfig, robotFactory);

        SimulationGUI video = new SimulationGUI(simulation);

        //new console which displays this simulation
        Console console = new Console(video);
        console.setVisible(true);
    }

    private Phenotype getPhenotypeForNetwork(NEATNetwork network)
    {
        if (isEvolvingMorphology() )
        {
            return new NEATMPhenotype((NEATMNetwork) network);
        }
        else
        {
            return new NEATPhenotype(network, sensorMorphology);
        }
    }

    public boolean isEvolvingMorphology()
    {
        return sensorMorphology == null;
    }

    public boolean isHyperNEATM()
    {
        return hyperNEATM;
    }

    public DescriptiveStatistics getPerformanceStatistics()
    {
        return performanceStats;
    }

    public DescriptiveStatistics getScoreStatistics()
    {
        return scoreStats;
    }

    public DescriptiveStatistics getSensorStatistics()
    {
        return sensorStats;
    }

    @Override
    public boolean shouldMinimize()
    {
        return false;
    }

    @Override
    public boolean requireSingleThreaded()
    {
        return false;
    }

}
