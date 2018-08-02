package za.redbridge.experiment;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import org.encog.Encog;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.neat.NEATNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;

import za.redbridge.experiment.HyperNEATM.HyperNEATMCODEC;
import za.redbridge.experiment.MultiObjective.MultiObjectiveEA;
import za.redbridge.experiment.MultiObjective.MultiObjectiveHyperNEATUtil;
import za.redbridge.experiment.MultiObjective.MultiObjectiveNEATMUtil;
import za.redbridge.experiment.NEATM.NEATMNetwork;
import za.redbridge.experiment.NEATM.NEATMPopulation;
import za.redbridge.experiment.NEATM.NEATMUtil;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.experiment.NEAT.NEATPopulation;
import za.redbridge.experiment.NEAT.NEATUtil;
import za.redbridge.experiment.HyperNEATM.SubstrateFactory;
import za.redbridge.simulator.config.ExperimentConfig;
import za.redbridge.simulator.config.SimConfig;


import static za.redbridge.experiment.Utils.isBlank;
import static za.redbridge.experiment.Utils.readObjectFromFile;

/**
 * Entry point for the experiment platform.
 * <p>
 * Created by jamie on 2014/09/09.
 */
public class Main
{

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    private static final double CONVERGENCE_SCORE = 110;

    public static void main(String[] args) throws IOException
    {
//        DaniString x = null;
//
//        ArrayList<DaniString> list1 = new ArrayList<>();
//        list1.add(x);
//
//        ArrayList<DaniString> list2 = new ArrayList<>();
//        list2.add(x);
//
//        list1.remove(x);
//
//        ArrayList<DaniString> list1 = new ArrayList<>();
//        list1.add(new DaniString(22));
//        ArrayList<DaniString> list2 = new ArrayList<>();
//        list2.add(list1.get(0));
//        list2.get(0).setAge(10);
//        System.out.println(list1.get(0).getAge());




        Args options = new Args();
        new JCommander(options, args);

        log.info(options.toString());
        //Loading in the config
        SimConfig simConfig;
        if (!isBlank(options.configFile))                   // if using config file
        {
            simConfig = new SimConfig(options.configFile);
        } else
        {
            simConfig = new SimConfig();
        }

        ScoreCalculator calculateScore =
                new ScoreCalculator(simConfig, options.trialsPerIndividual, null, options.hyperNEATM);

        if (!isBlank(options.genomePath))
        {
            NEATNetwork network = (NEATNetwork) readObjectFromFile(options.genomePath);
            calculateScore.demo(network);
            return;
        }


        final NEATPopulation population;
        if (!isBlank(options.populationPath))
        {
            population = (NEATPopulation) readObjectFromFile(options.populationPath);
        }
        else
        {
            if (options.hyperNEATM)
            {
                Substrate substrate = SubstrateFactory.createKheperaSubstrate(simConfig.getMinDistBetweenSensors(), simConfig.getRobotRadius());
                population = new NEATMPopulation(substrate, options.populationSize, options.multiObjective);
            }
            else
            {
                population = new NEATMPopulation(2, options.populationSize, options.multiObjective);
            }
            population.setInitialConnectionDensity(options.connectionDensity);
            population.reset();

            log.debug("Population initialized");
        }

        EvolutionaryAlgorithm train;
        if(options.hyperNEATM)
        {
            if(options.multiObjective)
            {
                train = MultiObjectiveHyperNEATUtil.constructNEATTrainer(population,calculateScore);
            }
            else
            {
                train = org.encog.neural.neat.NEATUtil.constructNEATTrainer(population, calculateScore);
                ((TrainEA)train).setCODEC(new HyperNEATMCODEC());
            }
        }
        else
        {

            if(options.multiObjective)
            {
                train = MultiObjectiveNEATMUtil.constructNEATTrainer(population, calculateScore);
            }
            else
            {
                train = NEATMUtil.constructNEATTrainer(population, calculateScore);
            }
        }
        //prevent elitist selection --> in future should use this for param tuning
        if(!options.multiObjective)
        {
            ((TrainEA)train).setEliteRate(0);
        }
        log.info("Available processors detected: " + Runtime.getRuntime().availableProcessors());
        if (options.threads > 0)
        {
            if(!options.multiObjective)
            {
                ((TrainEA)train).setThreadCount(options.threads);
            }
            else
            {
                ((MultiObjectiveEA)train).setThreadCount(options.threads);
            }
        }

        final StatsRecorder statsRecorder;
        if(options.multiObjective){
            statsRecorder = new MOStatsRecorder(train,calculateScore);
        }
        else{
            statsRecorder= new StatsRecorder(train, calculateScore);
        }


        for (int i = train.getIteration(); i < options.numGenerations; i++)
        {
            train.iteration();
            statsRecorder.recordIterationStats();

            if (train.getBestGenome().getScore() >= CONVERGENCE_SCORE)
            {
                log.info("Convergence reached at epoch " + train.getIteration());
                break;
            }
        }

        log.debug("Training complete");
        Encog.getInstance().shutdown();

        // #alex - save best network and run demo on it
        NEATNetwork bestPerformingNetwork = (NEATNetwork) train.getCODEC().decode(train.getBestGenome());   //extract best performing NN from the population
        calculateScore.demo(bestPerformingNetwork);
    }

    private static class Args
    {
        @Parameter(names = "-c", description = "Simulation config file to load")
        private String configFile = "config/bossConfig.yml";

        @Parameter(names = "-g", description = "Number of generations to train for")    // Jamie calls this 'iterations'
        private int numGenerations = 30;

        @Parameter(names = "-p", description = "Initial population size")
        private int populationSize = 5;

        @Parameter(names = "--trials", description = "Number of simulation runs per iteration (team lifetime)") // Jamie calls this 'simulationRuns' (and 'lifetime' in his paper)
        private int trialsPerIndividual = 1;

        @Parameter(names = "--conn-density", description = "Adjust the initial connection density"
                + " for the population")
        private double connectionDensity = 0.5;

        @Parameter(names = "--demo", description = "Show a GUI demo of a given genome")
        private String genomePath = null;

        @Parameter(names = "--HyperNEATM", description = "Using HyperNEATM")
        private boolean hyperNEATM = false;

        @Parameter(names = "--population", description = "To resume a previous experiment, provide"
                + " the path to a serialized population")
        private String populationPath = null;

        @Parameter(names = "--threads", description = "Number of threads to run simulations with."
                + " By default Runtime#availableProcessors() is used to determine the number of threads to use")
        private int threads = 0;

        // TODO description
        @Parameter(names = "--multi-objective", description = "Number of threads to run simulations with."
                + " By default Runtime#availableProcessors() is used to determine the number of threads to use")
        private boolean multiObjective = true;

        @Override
        public String toString()
        {
            return "Options: \n"
                    + "\tConfig file path: " + configFile + "\n"
                    + "\tNumber of simulation steps: " + numGenerations + "\n"
                    + "\tPopulation size: " + populationSize + "\n"
                    + "\tNumber of simulation tests per iteration: " + trialsPerIndividual + "\n"
                    + "\tInitial connection density: " + connectionDensity + "\n"
                    + "\tDemo network config path: " + genomePath + "\n"
                    + "\tHyperNEATM: " + hyperNEATM + "\n"
                    + "\tPopulation path: " + populationPath + "\n"
                    + "\tNumber of threads: " + threads;
        }
    }
}
