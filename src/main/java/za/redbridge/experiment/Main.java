package za.redbridge.experiment;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.encog.Encog;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.neat.NEATNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import za.redbridge.experiment.HyperNEATM.HyperNEATMCODEC;
import za.redbridge.experiment.HyperNEATM.HyperNEATMUtil;
import za.redbridge.experiment.HyperNEATM.SubstrateFactory;
import za.redbridge.experiment.MultiObjective.MultiObjectiveEA;
import za.redbridge.experiment.MultiObjective.MultiObjectiveHyperNEATUtil;
import za.redbridge.experiment.MultiObjective.MultiObjectiveNEATMUtil;
import za.redbridge.experiment.NEAT.NEATPopulation;
import za.redbridge.experiment.NEATM.NEATMPopulation;
import za.redbridge.experiment.NEATM.NEATMUtil;
import za.redbridge.experiment.SingleObjective.SingleObjectiveEA;
import za.redbridge.simulator.config.SimConfig;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static za.redbridge.experiment.Utils.isBlank;
import static za.redbridge.experiment.Utils.readObjectFromFile;

/**
 * Entry point for the experiment platform.
 * <p>
 * Created by jamie on 2014/09/09.
 */
public class Main
{
    private static final double CONVERGENCE_SCORE = 110;

    public static void main(String[] args) throws IOException
    {
        String date = new SimpleDateFormat("MMdd'T'HHmm").format(new Date());
        System.setProperty("log.name", "logback-"+date+".log");
        Utils.date = date;

        Logger log = LoggerFactory.getLogger(Main.class);

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

        String type = "";
        if (!isBlank(options.genomePath))
        {
            NEATNetwork network = (NEATNetwork) readObjectFromFile(options.genomePath);
            calculateScore.demo(network);
            return;
        }

        final NEATPopulation population;
        String popDirectory ="";
        if (!isBlank(options.populationPath))
        {
            population = (NEATPopulation) readObjectFromFile(options.populationPath);
            popDirectory = options.populationPath.split("results/")[1].split("/populations")[0];
        }
        else
        {
            if (options.hyperNEATM)
            {
                type = "HyperNEATM";
                Substrate substrate = SubstrateFactory.createKheperaSubstrate(simConfig.getMinDistBetweenSensors(), simConfig.getRobotRadius());
                population = new NEATMPopulation(substrate, options.populationSize, options.multiObjective);
            }
            else
            {
                type = "NEATM";
                population = new NEATMPopulation(2, options.populationSize, options.multiObjective);
            }
            population.setInitialConnectionDensity(options.connectionDensity);
            population.reset();

            log.debug("Population initialized");
        }

        EvolutionaryAlgorithm train;
        if (options.hyperNEATM)
        {
            if (options.multiObjective)
            {
                type = "MO-" + type;
                train = MultiObjectiveHyperNEATUtil.constructNEATTrainer(population, calculateScore, options.numGenerations);
            }
            else
            {
                train = HyperNEATMUtil.constructNEATTrainer(population, calculateScore);
                ((SingleObjectiveEA) train).setCODEC(new HyperNEATMCODEC());
            }
        }
        else    // if NEATM
        {
            if (options.multiObjective)
            {
                type = "MO-" + type;
                train = MultiObjectiveNEATMUtil.constructNEATTrainer(population, calculateScore, options.numGenerations);
            }
            else
            {
                train = NEATMUtil.constructNEATTrainer(population, calculateScore);
            }
        }
        //prevent elitist selection --> in future should use this for param tuning
        if (!options.multiObjective)
        {
            ((SingleObjectiveEA) train).setEliteRate(0.2);
        }
        log.info("Available processors detected: " + Runtime.getRuntime().availableProcessors());
        if (options.threads > 0)
        {
            if (!options.multiObjective)
            {
                ((SingleObjectiveEA) train).setThreadCount(options.threads);
            }
            else
            {
                ((MultiObjectiveEA) train).setThreadCount(options.threads);
            }
        }

        final StatsRecorder statsRecorder;

        if (options.multiObjective)
        {

            statsRecorder = new MOStatsRecorder(train, calculateScore, type, options.configFile,popDirectory);
        }
        else
        {
            statsRecorder = new StatsRecorder(train, calculateScore, type, options.configFile,popDirectory);
        }

        if(options.populationPath!=null){
            if(options.multiObjective){
                ((MultiObjectiveEA)train).firstIterationResume();
            }
            else{
                ((SingleObjectiveEA)train).firstIterationResume();
            }
        }

        for (int i = train.getIteration(); i < options.numGenerations; i++)
        {
            train.iteration();
            statsRecorder.recordIterationStats();

            /*
            // set covergence for single objective
            if(!options.multiObjective)
            {
                if (train.getBestGenome().getScore() >= CONVERGENCE_SCORE)
                {
                    log.info("Convergence reached at epoch " + train.getIteration());
                    break;
                }
            }
            */
        }

        log.debug("Training complete");
        Encog.getInstance().shutdown();

        // #alex - save best network and run demo on it
        //NEATNetwork bestPerformingNetwork = (NEATNetwork) train.getCODEC().decode(train.getBestGenome());   //extract best performing NN from the population
        //calculateScore.demo(bestPerformingNetwork);
    }

    public static class Args
    {
        @Parameter(names = "-c", description = "Simulation config file to load")
        //public static String configFile = "config/bossConfig.yml";
        public static String configFile = "config/ConfigSimple.yml";
        //private String configFile = "config/ConfigMedium.yml";
        // private String configFile = "config/ConfigDifficult.yml";

        @Parameter(names = "-g", description = "Number of generations to train for")    // Jamie calls this 'iterations'
        public static int numGenerations = 250;

        @Parameter(names = "-p", description = "Initial population size")
        public static int populationSize = 150;

        @Parameter(names = "--trials", description = "Number of simulation runs per iteration (team lifetime)")
        // Jamie calls this 'simulationRuns' (and 'lifetime' in his paper)
        public static int trialsPerIndividual = 5;

        @Parameter(names = "--conn-density", description = "Adjust the initial connection density"
                + " for the population")
        //NEAT
        public static double connectionDensity = 0.5;
        //HyperNEAT
        //private double connectionDensity = 0.9;

        @Parameter(names = "--demo", description = "Show a GUI demo of a given genome")
        public static String genomePath = null;

        @Parameter(names = "--HyperNEATM", description = "Using HyperNEATM")
        public static boolean hyperNEATM = false;

        @Parameter(names = "--population", description = "To resume a previous experiment, provide"
                + " the path to a serialized population")
        public static String populationPath =null;

        //private String populationPath = "/mnt/lustre/users/dnagar/experiment-...";

        //private String populationPath = "/mnt/lustre/users/afurman/experiment-...";

        @Parameter(names = "--threads", description = "Number of threads to run simulations with."
                + " By default Runtime#availableProcessors() is used to determine the number of threads to use")
        public static int threads = 0;

        @Parameter(names = "--multi-objective", description = "Using Multi-Objective NEAT/HyperNEAT"
                +" Based on NEAT-MODS")
        public static boolean multiObjective = false;

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
                    + "\tNumber of threads: " + threads + "\n"
                    + "\tMulti-objective: " + multiObjective;
        }
    }
}
