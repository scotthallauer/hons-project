package za.redbridge.experiment.MultiObjective;

import org.encog.Encog;
import org.encog.EncogShutdownTask;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLContext;
import org.encog.ml.MLMethod;
import org.encog.ml.ea.codec.GeneticCODEC;
import org.encog.ml.ea.codec.GenomeAsPhenomeCODEC;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.opp.EvolutionaryOperator;
import org.encog.ml.ea.opp.OperationList;
import org.encog.ml.ea.opp.selection.SelectionOperator;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.rules.RuleHolder;
import org.encog.ml.ea.score.AdjustScore;
import org.encog.ml.ea.sort.*;
import org.encog.ml.ea.species.SingleSpeciation;
import org.encog.ml.ea.species.Speciation;
import org.encog.ml.ea.species.Species;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.ml.genetic.GeneticError;
import org.encog.util.concurrency.MultiThreadable;
import za.redbridge.experiment.MultiObjective.Comparator.DistanceComparator;
import za.redbridge.experiment.MultiObjective.Comparator.MaximisingObjectiveComparator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MultiObjectiveEA implements EvolutionaryAlgorithm, MultiThreadable,
        EncogShutdownTask, Serializable
{

    /**
     * The thread pool executor.
     */
    private ExecutorService taskExecutor;

    /**
     * The number of times to try certian operations, so an endless loop does
     * not occur.
     */
    private int maxTries = 5;

    /**
     * Should exceptions be ignored.
     */
    private boolean ignoreExceptions;

    /**
     * The validation mode.
     */
    private boolean validationMode;

    /**
     * The selection operator.
     */
    private SelectionOperator selection;

    /**
     * Holds rewrite and constraint rules.
     */
    private RuleHolder rules;

    /**
     * The speciation method.
     */
    private Speciation speciation = new SingleSpeciation();

    /**
     * The iteration number.
     */
    private int iteration;

    /**
     * The genome comparator.
     */
    private GenomeComparator selectionComparator;

    /**
     * The best ever genome.
     */
    private Genome bestGenome;

    /**
     * The genome comparator.
     */
    private GenomeComparator bestComparator;

    /**
     * The operators. to use.
     */
    private final OperationList operators = new OperationList();


    /**
     * The number of requested threads.
     */
    private int threadCount;

    /**
     * The mutation to be used on the top genome. We want to only modify its
     * weights.
     */
    private EvolutionaryOperator champMutation;

    /**
     * The CODEC to use to convert between genome and phenome.
     */
    private GeneticCODEC codec = new GenomeAsPhenomeCODEC();

    /**
     * The population.
     */
    private Population population;

    /**
     * The score calculation function.
     */
    private final CalculateScore scoreFunction;

    /**
     * The population from the previous iteration.
     */
    private List<Genome> parentPopulation;

    /**
     * The population for the next iteration.
     */
    private List<Genome> offspringPopulation;

    /**
     * The actual thread count.
     */
    private int actualThreadCount = -1;


    /**
     * Create a trainer for a score function.
     * @param thePopulation The population.
     * @param theScoreFunction The score function.
     */
    public MultiObjectiveEA(Population thePopulation, CalculateScore theScoreFunction)
    {
        this.population = thePopulation;
        this.scoreFunction = theScoreFunction;

        // set the score compare method
        if (theScoreFunction.shouldMinimize())
        {
            this.selectionComparator = new MinimizeAdjustedScoreComp();
            this.bestComparator = new MinimizeScoreComp();
        } else
        {
            this.selectionComparator = new MaximizeAdjustedScoreComp();
            this.bestComparator = new MaximizeScoreComp();
        }

        // set the iteration
        for (final Species species : thePopulation.getSpecies())
        {
            for (final Genome genome : species.getMembers())
            {
                setIteration(Math.max(getIteration(),
                        genome.getBirthGeneration()));
            }
        }

        // Set a best genome, just so it is not null.
        // We won't know the true best genome until the first iteration.
        if( this.population.getSpecies().size()>0 && this.population.getSpecies().get(0).getMembers().size()>0 ) {
            this.bestGenome = this.population.getSpecies().get(0).getMembers().get(0);
        }
    }


    @Override
    public void performShutdownTask()
    {
        finishTraining();
    }

    /**
     * Add an operation.
     *
     * @param probability The probability of using this operator.
     * @param opp
     */
    @Override
    public void addOperation(final double probability, final EvolutionaryOperator opp)
    {
        getOperators().add(probability, opp);
        opp.init(this);
    }

    /**
     * Add a score adjuster. Score adjusters are used to adjust the adjusted
     * score of a genome. This allows bonuses and penalties to be applied for
     * desirable or undesirable traits.
     * desirable or undesirable trits.
     *
     * @param scoreAdjust The score adjustor to add.
     */
    @Override
    public void addScoreAdjuster(AdjustScore scoreAdjust)
    {

    }

    /**
     * Calculate the score for a genome.
     *
     * @param g The genome to calculate the score for.
     */
    @Override
    public void calculateScore(Genome g)
    {
        // try rewrite
        this.rules.rewrite(g);

        // decode
        final MLMethod phenotype = getCODEC().decode(g);
        double score;

        // deal with invalid decode
        if (phenotype == null)
        {
            if (getBestComparator().shouldMinimize())
            {
                score = Double.POSITIVE_INFINITY;
            }
            else
            {
                score = Double.NEGATIVE_INFINITY;
            }
        }
        else
        {
            if (phenotype instanceof MLContext)
            {
                ((MLContext) phenotype).clearContext();
            }
            score = getScoreFunction().calculateScore(phenotype);
        }

        // now set the scores
        g.setScore(score);
        g.setAdjustedScore(score);
    }

    /**
     * Called when training is finished. This allows the EA to properly shut
     * down.
     */
    @Override
    public void finishTraining()
    {
        // wait for threadpool to shutdown
        if (this.taskExecutor != null) {
            this.taskExecutor.shutdown();
            try {
                this.taskExecutor.awaitTermination(Long.MAX_VALUE,
                        TimeUnit.MINUTES);
            } catch (final InterruptedException e) {
                throw new GeneticError(e);
            } finally {
                this.taskExecutor = null;
                Encog.getInstance().removeShutdownTask(this);
            }
        }
    }

    /**
     * Get the comparator that is used to choose the "true best" genome. This
     * uses the real score, and not the adjusted score.
     *
     * @return The best comparator.
     */
    @Override
    public GenomeComparator getBestComparator()
    {
        return this.bestComparator;
    }

    /**
     * @return The current best genome. This genome is safe to use while the EA
     * is running. Genomes are not modified. They simply produce
     * "offspring".
     */
    @Override
    public Genome getBestGenome() { return this.bestGenome;}

    /**
     * @return The CODEC that is used to transform between genome and phenome.
     */
    @Override
    public GeneticCODEC getCODEC()
    {
        return this.codec;
    }

    /**
     * @return The current score. This value should either be minimized or
     * maximized, depending on the score function.
     */
    @Override
    public double getError()
    {
        // do we have a best genome, and does it have an error?
        if (this.bestGenome != null)
        {
            double err = this.bestGenome.getScore();
            if( !Double.isNaN(err) )
            {
                return err;
            }
        }
        // otherwise, assume the worst!
        if (getScoreFunction().shouldMinimize())
        {
            return Double.POSITIVE_INFINITY;
        }
        else
        {
            return Double.NEGATIVE_INFINITY;
        }
    }

    /**
     * @return The current iteration number. Also sometimes referred to as
     * generation or epoch.
     */
    @Override
    public int getIteration()
    {
        return this.iteration;
    }

    /**
     * @return The maximum size an individual genome can be. This is an
     * arbitrary number defined by the genome. Lower numbers are less
     * complex.
     */
    @Override
    public int getMaxIndividualSize()
    {
        return this.population.getMaxIndividualSize();
    }

    /**
     * @return The maximum number to try certain genetic operations. This
     * prevents endless loops.
     */
    @Override
    public int getMaxTries()
    {
        return this.maxTries;
    }

    /**
     * @return The operators.
     */
    @Override
    public OperationList getOperators()
    {
        return this.operators;
    }

    /**
     * @return The population.
     */
    @Override
    public Population getPopulation()
    {
        return this.population;
    }

    /**
     * @return The rules holder, contains rewrite and constraint rules.
     */
    @Override
    public RuleHolder getRules()
    {
        return this.rules;
    }

    /**
     * @return The score adjusters. This allows bonuses and penalties to be
     * applied for desirable or undesirable traits.
     */
    @Override
    public List<AdjustScore> getScoreAdjusters()
    {
        return null;
    }   // todo: check if this is needed

    /**
     * @return The score function.
     */
    @Override
    public CalculateScore getScoreFunction()
    {
        return this.scoreFunction;
    }

    /**
     * @return The selection operator. Used to choose genomes.
     */
    @Override
    public SelectionOperator getSelection()
    {
        return this.selection;
    }

    /**
     * Get the comparator that is used to choose the "best" genome for
     * selection, as opposed to the "true best". This uses the adjusted score,
     * and not the score.
     *
     * @return The selection comparator.
     */
    @Override
    public GenomeComparator getSelectionComparator()
    {
        return this.selectionComparator;
    }

    /**
     * @return True if exceptions that occur during genetic operations should be
     * ignored.
     */
    @Override
    public boolean getShouldIgnoreExceptions()
    {
        return this.ignoreExceptions;
    }

    /**
     * @return The speciation method.
     */
    @Override
    public Speciation getSpeciation()
    {
        return this.speciation;
    }

    /**
     * @return True if any genome validators should be applied.
     */
    @Override
    public boolean isValidationMode()
    {
        return this.validationMode;
    }

    private void setNonDominationAndCrowdDists()
    {
        HashMap<MultiObjectiveGenome, ArrayList<MultiObjectiveGenome>> S = new HashMap<>();
        HashMap<MultiObjectiveGenome, Integer> n = new HashMap<>();  // number of individuals that dominate p
        ArrayList<ArrayList<MultiObjectiveGenome>> Fronts = new ArrayList<>();
        ArrayList<Species> SelectedSpecies = new ArrayList<Species>();

        for(Species species : population.getSpecies())
        {
            for (Genome pp : species.getMembers())
            {
                MultiObjectiveGenome p = (MultiObjectiveGenome)pp;
                System.out.println(p.getScoreVector().get(0) + " ... "+p.getScoreVector().get(1));
            }
        }
        Fronts.add(new ArrayList<MultiObjectiveGenome>());
        for(Species species : population.getSpecies())
        {
            for(Genome pp : species.getMembers())
            {
                MultiObjectiveGenome p = (MultiObjectiveGenome)pp;
                S.put(p, new ArrayList<MultiObjectiveGenome>());
                n.put(p, 0);
                for(Species species2 : population.getSpecies())
                {
                    for(Genome qq : species2.getMembers())
                    {
                        MultiObjectiveGenome q = (MultiObjectiveGenome)qq;
                        if(checkADominatesB(p,q))
                        {
                            S.get(p).add(q);
                        }
                        else if (checkADominatesB(q,p))
                        {
                            n.put(p,n.get(p)+1);
                        }
                    }
                }
                if(n.get(p).equals(0))   // if p dominated by no one
                {

                    p.setRank(0);    // then p belongs to the 1st Pareto front
                    Fronts.get(0).add(p);   // add p to first front

                }
            }
        }

        // first iteration now finished - everyone scored


        int i = 0; // initialise Front counter to 0

        while(!(Fronts.get(i).size()==0))
        {
            Fronts.set(i,crowdingDistance(Fronts.get(i))); //sorts on crowding distance

            ArrayList<MultiObjectiveGenome> Q = new ArrayList<>();  // will store individuals in Fronts[i+1]

            for(MultiObjectiveGenome p: Fronts.get(i))
            {
                //Selected Species add
                if(!SelectedSpecies.contains(p.getSpecies())) {
                    SelectedSpecies.add(p.getSpecies());
                 }

                for(MultiObjectiveGenome q: S.get(p))   // for each invidiual q that is dominated by p
                {
                    n.put(q,n.get(q)-1);    // decrement # of individuals that dominate q
                    if(n.get(q).equals(0))     // if q is only dominated by the first front
                    {
                        q.setRank(i+1)              ;      // then it belongs in the second front
                        Q.add(q);
                    }
                }
            }

            Fronts.add(Q); // update the next front to be Q

            i++;                                     // increment front counter
        }
       // System.out.println("i "+ i);
       // System.out.println("fronts"+Fronts.size());

    }

    private ArrayList<MultiObjectiveGenome> crowdingDistance(ArrayList<MultiObjectiveGenome> front){
        int l = front.size();
        for(MultiObjectiveGenome i : front)
        {
            i.setDistance(0.0);
        }
        MaximisingObjectiveComparator<MultiObjectiveGenome> comparator = new MaximisingObjectiveComparator<>();
        for(int i =0;i<2;i++){
            comparator.setObjectiveIndex(i);
            Collections.sort(front,comparator);
            front.get(0).setDistance(Double.POSITIVE_INFINITY);
            front.get(l-1).setDistance(Double.POSITIVE_INFINITY);
            for(int j =1;j<l-1;j++){
                front.get(j).setDistance(front.get(j).getDistance()+(front.get(j+1).getScoreVector().get(i)-front.get(j-1).getScoreVector().get(i)));
            }

        }

        Collections.sort(front, new DistanceComparator<MultiObjectiveGenome>());
        return front;



    }

    private boolean checkADominatesB(MultiObjectiveGenome a, MultiObjectiveGenome b) // check if a dominates b
    {



        if(a.getScoreVector().get(0) >= b.getScoreVector().get(0) && a.getScoreVector().get(1) >= b.getScoreVector().get(1))
        {
            if(a.getScoreVector().get(0) > b.getScoreVector().get(0) || a.getScoreVector().get(1) > b.getScoreVector().get(1))
            {
                return true; // a dominates b
            }
        }
        return false;
    }

    /**
     * The first iteration of Multi-Objective NEAT. Also determines number of threads to use.
     */
    public void firstIteration()
    {
        // init speciation
        getSpeciation().init(this);

        // Threads - find out how many threads to use
        if (this.getThreadCount() == 0) this.actualThreadCount = Runtime.getRuntime().availableProcessors();
        else this.actualThreadCount = this.getThreadCount();

        // Score the initial (parent) population
        final MultiObjectiveParallelScore pscore = new MultiObjectiveParallelScore(getPopulation(),
                getCODEC(), new ArrayList<AdjustScore>(), getScoreFunction(),
                this.actualThreadCount);
        pscore.setThreadCount(this.actualThreadCount);
        pscore.process();
        this.actualThreadCount = pscore.getThreadCount();

        // start up the thread pool
        if (this.actualThreadCount == 1)
        {
            this.taskExecutor = Executors.newSingleThreadScheduledExecutor();
        }
        else
        {
            this.taskExecutor = Executors.newFixedThreadPool(this.actualThreadCount);
        }

        // register for shutdown
        Encog.getInstance().addShutdownTask(this);

        // just pick the first genome with a valid score as best, it will be
        // updated later.
        // also most populations are sorted this way after training finishes
        // (for reload)
        // if there is an empty population, the constructor would have blow
        final List<Genome> list = getPopulation().flatten();

        int idx = 0;
        do
        {
            this.bestGenome = list.get(idx++);
        } while (idx < list.size()
                && (Double.isInfinite(this.bestGenome.getScore()) || Double
                .isNaN(this.bestGenome.getScore())));

        getPopulation().setBestGenome(this.bestGenome);

        // speciate
        final List<Genome> genomes = getPopulation().flatten();
        this.speciation.performSpeciation(genomes);

        // purge invalid genomes
        this.population.purgeInvalidGenomes();

        // Set Non-Domination ranks and Crowding Distance scores
        setNonDominationAndCrowdDists();


        // Set final scores



        // Speciation

        // done! (parentPopulation set)
    }

    /**
     * Perform a training iteration. Also called generations or epochs.
     */
    @Override
    public void iteration()
    {
        firstIteration();


        // Create offspringPopulation
            // offspringPopulation = mutateAndCrossover(parentPopulation)

        // Create combinedPopulation
            // combinedPopulation = offspringPopulation + parentPopulation

        // Evaluate combinedPopulation
            // compute each individual's performance vector (1 score for each objective)

        // speciate(combinedPopulation)

        // Selection Phase 1:

            // nonDominationSort(combinedPopulation)

            // setCrowdingDistances(combinedPopulation)

            // we now have a way of ranking species in combinedPopulation

            // now select the best species from combinedPopulation for Phase 2
                // setOfBestSpecies = selectBestSpecies(combinedPopulation)

        // Selection Phase 2:

            // Choose N best individuals from the set of best species
                // NChosenIndividuals = serialSpeciesProgression(setOfBestSpecies)

            // Set parent population to these N selected individuals
                // parentPopulation = the N selected individuals


    }

    /**
     * Set the comparator that is used to choose the "true best" genome. This
     * uses the real score, and not the adjusted score.
     *
     * @param bestComparator The best comparator.
     */
    @Override
    public void setBestComparator(final GenomeComparator bestComparator)
    {
        this.bestComparator = bestComparator;
    }

    /**
     * Set the population.
     *
     * @param thePopulation The population.
     */
    @Override
    public void setPopulation(final Population thePopulation)
    {
        this.population = population;
    }

    /**
     * Set the rules holder to use.
     *
     * @param rules The rules holder.
     */
    @Override
    public void setRules(final RuleHolder rules)
    {
        this.rules = rules;
    }

    /**
     * Set the selection operator.
     *
     * @param selection The selection operator.
     */
    @Override
    public void setSelection(final SelectionOperator selection)
    {
        this.selection = selection;
    }

    /**
     * Set the comparator that is used to choose the "best" genome for
     * selection, as opposed to the "true best". This uses the adjusted score,
     * and not the score.
     *
     * @param selectionComparator The selection comparator.
     */
    @Override
    public void setSelectionComparator(final GenomeComparator selectionComparator)
    {
        this.selectionComparator = selectionComparator;
    }

    /**
     * Determines if genetic operator exceptions should be ignored.
     *
     * @param b True if exceptions should be ignored.
     */
    @Override
    public void setShouldIgnoreExceptions(final boolean b)
    {
        this.ignoreExceptions = b;
    }

    /**
     * Set the speciation method.
     *
     * @param speciation The speciation method.
     */
    @Override
    public void setSpeciation(final Speciation speciation)
    {
        this.speciation = speciation;
    }

    /**
     * Determine if the genomes should be validated. This takes more time but
     * can help isolate a problem.
     *
     * @param validationMode True, if validation mode is enabled.
     */
    @Override
    public void setValidationMode(final boolean validationMode)
    {
        this.validationMode = validationMode;
    }

    /**
     * @return The number of threads to use, 0 to automatically
     * determine based on core count.
     */
    @Override
    public int getThreadCount()
    {
        return this.threadCount;
    }

    /**
     * Set the number of threads to use.
     *
     * @param numThreads The number of threads to use, or zero to
     *                   automatically determine based on core count.
     */
    @Override
    public void setThreadCount(final int numThreads)
    {
        this.threadCount = numThreads;
    }

    /**
     * Set the CODEC to use.
     *
     * @param theCodec
     *            The CODEC to use.
     */
    public void setCODEC(final GeneticCODEC theCodec)
    {
        this.codec = theCodec;
    }

    /**
     * @param champMutation
     *            the champMutation to set
     */
    public void setChampMutation(final EvolutionaryOperator champMutation) {
        this.champMutation = champMutation;
    }

    /**
     * Set the current iteration number.
     *
     * @param iteration
     *            The iteration number.
     */
    public void setIteration(final int iteration)
    {
        this.iteration = iteration;
    }

}
