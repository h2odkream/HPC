#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define ALFA 0.5
#define LAMBDA 0.1
#define BETA_MAX 1.0
#define BETA_MIN 0.0
#define INITIAL_LOUDNESS 1.0
#define DUMP_DIR "./dump"

typedef struct Bat {
    double pulse_rate;
    double loudness;
    double fitness;
    double frequency;
    double *position;
    double *velocity;
} Bat;

enum { N = 624 };        // length of state vector
enum { M = 397 };        // period parameter
unsigned long state[N];  // internal state
unsigned long *pNext;    // next value to get from state
int left;          	 // number of values left before reload needed
unsigned long MT_randInt(unsigned long n);
unsigned long randInt();
void reload();
unsigned long twist(unsigned long m, unsigned long s0, unsigned long s1);
unsigned long hiBit(unsigned long u);
unsigned long loBit(unsigned long u);
unsigned long loBits(unsigned long u);
unsigned long mixBits(unsigned long u, unsigned long v );
void MT_seed();
unsigned long MT_hash(time_t t, clock_t c);
void MT_seedfinal(unsigned long oneSeed);
void MT_initialize(unsigned long seed);
float MT_randfloat();
double MT_randExc(const double  *n );


const int LOG_OBJECTIVE_ENABLED=1;

extern int bats_count;
extern int dimensions;
extern int iterations;
extern int evaluation_function;
void logger(int destination, char *fmt, ...);
char* get_function_name(int index);

int boundry_max;
int BOUNDRY_MIN;
int FREQUENCY_MIN;
int FREQUENCY_MAX;
int FIRST_SEED;
int SECOND_SEED;

int BOUNDRY_SCAPE_COUNT = 0;
int BOUNDRY_COUNT = 0;


double (*objective_function)(double[], int);
double fitness_average(struct Bat bats[]);
double calc_loudness_average(struct Bat *bats);
void local_search(struct Bat *bat, struct Bat *best, double loudness_average);
void update_position(struct Bat *bat);
void update_velocity(struct Bat *bat, struct Bat *best);
void force_boundry_on_vector(double vector[]);
void force_boundry_on_value(double* value);
void log_objective(struct Bat *best, struct Bat *all_bats);

FILE *LOG_OBJECTIVE_FILE;
FILE *LOG_SCALAR_ATRIBUTES_FILE;
FILE *LOG_VECTOR_ATRIBUTES_FILE;
int RUN_TIME;
double my_rand(double inferior, double superior)
{
  double result = (double)inferior + ((superior - inferior)*MT_randInt(RAND_MAX)/(RAND_MAX+1.0));

    return result;
}


double shuber (double solution[], int dimensions)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < dimensions; i++) {
        sum += -sin(2.0*solution[i]+1.0)
            -2.0*sin(3.0*solution[i]+2.0)
            -3.0*sin(4.0*solution[i]+3.0)
            -4.0*sin(5.0*solution[i]+4.0)
            -5.0*sin(6.0*solution[i]+5.0);
    }
}


double sphere (double *solution, int dimensions)
{
    double total = 0;
	int i;
    for (i = 0; i < dimensions; i++) {
        total+= solution[i] * solution[i];
    }

    return total;
}

double rosenbrock(double solution[], int dimensions)
{
    double total = 0;
    int i;
    for (i = 0; i < dimensions-1; i++)
    {
        total=total+100.*pow((solution[i+1] - pow(solution[i],2.)),2) + pow((1. - solution[i]),2);
    }

    return total;
}


double schwefel(double solution[], int dimensions)
{
    double aux = 0;
    int i;
    for (i=0;i<dimensions;i++)
    {
        aux += solution[i]*sin(sqrt(fabs(solution[i])));
    }
    return(-1*aux/dimensions);
}

double rastringin(double solution[], int dimensions)
{
    double total = 0;
	int i;
    for(i=0;i<dimensions;i++)
    {
        total=total+(pow(solution[i],(double)2)-10*cos(2*M_PI*solution[i])+10);
    }

    return total;
}

double griewank(double solution[], int dimensions)
{
    double total = 0;

    double top1=0;
    double top2=1;
	int i;
    for(i=0;i<dimensions;i++)
    {
        top1=top1+pow((solution[i]),(double)2);
        top2=top2*cos((((solution[i])/sqrt((double)(i+1)))*M_PI)/180);
    }
    total=(1/(double)4000)*top1-top2+1;

    return total;
}

double ackley(double solution[], int dimensions)
{
    int i;
    double aux, aux1, result;

    for (i = 0; i < dimensions; i++)
    {
        aux += solution[i]*solution[i];
    }
    for (i = 0; i < dimensions; i++)
    {
        aux1 += cos(2.0*M_PI*solution[i]);
    }

    result = -20.0*(exp(-0.2*sqrt(1.0/(float)dimensions*aux)))-exp(1.0/(float)dimensions*aux1)+20.0+exp(1);

    return result;
}

void bat_copy(struct Bat *from, struct Bat *to)
{
    memcpy(to, from, sizeof(struct Bat));
}

struct Bat get_best(struct Bat *bats, struct Bat *best)
{
    double current_best_val;
    int best_indice;

    current_best_val = bats[0].fitness;
    best_indice = 0;
    int i;
    for (i = 0; i < bats_count; i++) {
        if (bats[i].fitness < current_best_val) {
            current_best_val = bats[i].fitness;
            best_indice = i;
        }
    }
    bat_copy(&bats[best_indice], best);
}


void logger(int destination, char *fmt, ...)
{
    char formatted_string[6666];

    va_list argptr;
    va_start(argptr,fmt);
    vsprintf(formatted_string, fmt, argptr);
    va_end(argptr);

    if (destination == LOG_OBJECTIVE)
        fprintf(LOG_OBJECTIVE_FILE,"%s",formatted_string);
    else if (destination == LOG_STDOUT)
        printf("%s",formatted_string);
}




void bat_stdout(struct Bat *bat, int dimensions)
{
    double position_average =  0;
    int i;
    for (i = 0; i < dimensions; i++) {
        position_average+=bat->position[i];
    }
    position_average/=dimensions;
    printf("ITERATIONS: %d\n", iterations);
    printf("BATS_COUNT: %d\n", bats_count);
    printf("DIMENSIONS: %d\n", dimensions);
    printf("POPULATION: %d\n", bats_count);
    logger(LOG_STDOUT, "Fitness E: %E\n", bat->fitness);
}

struct Bat *bat_factory()
{
    Bat *bat;
    bat = malloc (sizeof *bat);

    bat->loudness = INITIAL_LOUDNESS;

    bat->velocity = (double *) malloc(sizeof(bat->position) * dimensions);
    bat->position = (double *) malloc(sizeof(bat->position) * dimensions);
	int j;
    for (j = 0; j < dimensions; j++) {
        bat->velocity[j] = my_rand(BOUNDRY_MIN, boundry_max);
        bat->position[j] = my_rand(BOUNDRY_MIN, boundry_max);
    }

    bat->fitness = fabs(objective_function(bat->position, dimensions));

    return bat;
}


void log_objective(struct Bat *best, struct Bat *all_bats)
{
    double average_fitness = fitness_average(all_bats);
    logger(LOG_OBJECTIVE, "%E %E\n", best->fitness, average_fitness);
}


void force_boundry_on_value(double* value)
{
    BOUNDRY_COUNT++;
    if (*value > boundry_max) {
        *value = boundry_max;
        BOUNDRY_SCAPE_COUNT++;
        return;
    }
    if (*value < BOUNDRY_MIN) {
        *value = BOUNDRY_MIN;
        BOUNDRY_SCAPE_COUNT++;
    }
}

void force_boundry_on_vector(double vector[])
{
	int i;
    for (i = 0; i < dimensions; i++ ) {
        force_boundry_on_value(&vector[i]);
    }
}

void update_velocity(struct Bat *bat, struct Bat *best)
{
	int i;
    for (i = 0; i < dimensions; ++i) {
        bat->velocity[i]+= (bat->position[i] - best->position[i]) * bat->frequency;
        force_boundry_on_value(&bat->velocity[i]);
    }
}

void update_position(struct Bat *bat)
{
	int i;
    for (i = 0; i < dimensions; ++i) {
        bat->position[i] += bat->velocity[i];

        force_boundry_on_value(&bat->position[i]);
    }
}

void local_search(struct Bat *bat, struct Bat *best, double loudness_average)
{
	int i;
    for (i = 0; i < dimensions; i++ ) {
        bat->position[i] = best->position[i] + loudness_average * my_rand(-1.0, 1.0);
    }
}

double calc_loudness_average(struct Bat *bats)
{
    double total = 0;

	int i;
    for(i=0;i<bats_count;i++) {
        total+= bats[i].loudness;
    }

    return total / bats_count;
}

double fitness_average(struct Bat bats[])
{
    double result = 0;
	int i;
    for(i = 0; i < bats_count; i++) {
        result+= bats[i].fitness;
    }

    return result / bats_count;
}


double bat_loudness_get(struct Bat *bat)
{
    return bat->loudness;
}

double bat_fitness_get(struct Bat *bat)
{
    return bat->fitness;
}

double bat_pulse_rate_get(struct Bat *bat)
{
    return bat->pulse_rate;
}

struct Bat* get_worst(struct Bat *bats)
{
    double current_worst_val;
    double current_val;

    current_val = current_worst_val = bats[0].fitness;
    int worst_indice = 0;
    int i;
    for(i = 0; i < bats_count; i++) {
        current_val = bats[i].fitness;
        if (current_worst_val <  current_val) {
            current_worst_val = current_val;
            worst_indice = i;
        }
    }

    return &bats[worst_indice];
}


void initialize_function(int evaluation_function)
{
    switch(evaluation_function) {
        case SPHERE:
            BOUNDRY_MIN = 0.0;
            boundry_max = 100.0;
            objective_function = &sphere;
            break;
        case RASTRINGIN:
            BOUNDRY_MIN = -5.12;
            boundry_max = 5.12;
            objective_function = &rastringin;
            break;
        case GRIEWANK:
            BOUNDRY_MIN = -600.0;
            boundry_max = 600.0;
            objective_function = &griewank;
            break;
        case ACKLEY:
            BOUNDRY_MIN = -32.0;
            boundry_max = 32.0;
            objective_function = &ackley;
            break;
        case SHUBER:
            BOUNDRY_MIN = -100.0;
            boundry_max = 100.0;
            objective_function = &shuber;
            break;
        case SCHWEFEL:
            BOUNDRY_MIN = -500.0;
            boundry_max = 500.0;
            objective_function = &schwefel;
            break;
        case ROSENBROOK:
            BOUNDRY_MIN = -30.0;
            boundry_max = 30.0;
            objective_function = &rosenbrock;
            break;
    }
}

void bat_run(void)
{
    struct Bat bats[bats_count];
    struct Bat *best;
    struct Bat *candidate;
    double best_result,average_result,worst_result;

    char fileName[100];

    if (LOG_OBJECTIVE_ENABLED) {
        sprintf(fileName, "%s/%i-objective", DUMP_DIR, time(NULL));
        LOG_OBJECTIVE_FILE = fopen(fileName,"w");
        if (LOG_OBJECTIVE_FILE == NULL)
        {
            printf("Error opening file %s !\n", fileName);
            exit(1);
        }
        printf("Objective log: %s\n", fileName);
    }

    initialize_function(evaluation_function);

    FREQUENCY_MIN=BOUNDRY_MIN;
    FREQUENCY_MAX=boundry_max;

    MT_seed();

    best = bat_factory();
    candidate = bat_factory();
	int i;
    for(i = 0; i < bats_count; i++) {
        bats[i] = *bat_factory();
    }
	int iteration, j;
    for(iteration = 0; iteration < iterations ; ++iteration) {
        for (j = 0; j < bats_count; ++j) {

            double beta = my_rand(BETA_MIN, BETA_MAX);
            bats[j].frequency = FREQUENCY_MIN + (FREQUENCY_MAX - FREQUENCY_MIN) * beta;

            update_velocity(&bats[j], best);
            bat_copy(&bats[j], candidate);

            update_position(candidate);

            if (my_rand(0.0,1.0) < candidate->pulse_rate) {
                local_search(candidate, best, calc_loudness_average(bats));
            }

            int dimension = my_rand(0, dimensions-1);
            //bat->position[dimension] = bat->position[dimension] * my_rand(0.0,1.0);
            force_boundry_on_vector(candidate->position);

            bats[j].fitness = fabs(objective_function(bats[j].position, dimensions));
            candidate->fitness = fabs(objective_function(candidate->position, dimensions));
            if (my_rand(0.0,1.0) < bats[j].loudness && candidate->fitness < bats[j].fitness) {
                bat_copy(candidate, &bats[j]);
                bats[j].fitness = candidate->fitness;
                bats[j].pulse_rate = 1 - exp(-LAMBDA*iteration);

                bats[j].loudness = INITIAL_LOUDNESS*pow(ALFA, iteration);
            }
        }
        get_best(bats, best);
        if (LOG_OBJECTIVE_ENABLED) {
            log_objective(best, bats);
        }
    }

    bat_stdout(best,dimensions);

    if (LOG_OBJECTIVE_ENABLED) {
        fclose(LOG_OBJECTIVE_FILE);
    }
}


unsigned long MT_randInt(unsigned long n)      // inteiro entre [0,n] para n < 2^32
{
	unsigned long used = n;
	used |= used >> 1;
	used |= used >> 2;
	used |= used >> 4;
	used |= used >> 8;
	used |= used >> 16;
	unsigned long i;
	do{
		i = randInt() & used;
	}while( i > n );
	return i;
}
unsigned long randInt()
{
	register unsigned long s1;

	if( left == 0 ) reload();
	--left;

	s1 = *pNext++;
	s1 ^= (s1 >> 11);
	s1 ^= (s1 <<  7) & 0x9d2c5680UL;
	s1 ^= (s1 << 15) & 0xefc60000UL;

	return ( s1 ^ (s1 >> 18) );
}

void reload()
{
	register unsigned long *p = state;
	register int i;
	for( i = N - M; i--; ++p )
		*p = twist( p[M], p[0], p[1] );
	for( i = M; --i; ++p )
		*p = twist( p[M-N], p[0], p[1] );
	*p = twist( p[M-N], p[0], state[0] );

	left = N, pNext = state;
}
unsigned long twist(unsigned long m, unsigned long s0, unsigned long s1 )
{
	return m ^ (mixBits(s0,s1)>>1) ^ (-loBit(s1) & 0x9908b0dfUL);
}


void MT_seed()
{
	FILE* urandom = fopen( "/dev/urandom", "rb" );
	/*if( urandom )
	{
		unsigned long bigSeed[N];
		register unsigned long *s = bigSeed;
		register int i=N;
		register bool success = true;
		while( success && i-- )
			success = fread( s++, sizeof(unsigned long), 1, urandom );
		fclose(urandom);
		if( success ) { seed( bigSeed, N );  return; }
	}*/
	MT_seedfinal( MT_hash( time(NULL), clock() ) );
}

unsigned long MT_hash(time_t t, clock_t c)
{
	size_t i, j;
	static unsigned long  differ = 0;

	unsigned long  h1 = 0;
	unsigned char *p = (unsigned char *) &t;
	for(i = 0; i < sizeof(t); ++i)

	{
		h1 *= UCHAR_MAX + 2U;
		h1 += p[i];
	}
	unsigned long  h2 = 0;
	p = (unsigned char *) &c;
	for(j = 0; j < sizeof(c); ++j)
	{
		h2 *= UCHAR_MAX + 2U;
		h2 += p[j];
	}
	return ( h1 + differ++ ) ^ h2;
}

void MT_seedfinal(unsigned long oneSeed)
{
	MT_initialize(oneSeed);
	reload();
}

void MT_initialize(unsigned long seed)
{
	register unsigned long *s = state;
	register unsigned long *r = state;
	register int i = 1;
	*s++ = seed & 0xffffffffUL;
	for( ; i < N; ++i )
	{
		*s++ = ( 1812433253UL * ( *r ^ (*r >> 30) ) + i ) & 0xffffffffUL;
		r++;
	}
}

float MT_randfloat()
{
	return (float)(randInt()) * (1.0/4294967295.0);
}

double MT_rand()
    { return (double) (randInt()) * (1.0/4294967296.0);
    }

double MT_randExc(const double  *n )
    { return MT_rand() * *n;
    }

unsigned long hiBit(unsigned long u) { return u & 0x80000000UL; }
unsigned long loBit(unsigned long u) { return u & 0x00000001UL; }
unsigned long loBits(unsigned long u){ return u & 0x7fffffffUL; }
unsigned long mixBits(unsigned long u, unsigned long v ) { return hiBit(u) | loBits(v); }
