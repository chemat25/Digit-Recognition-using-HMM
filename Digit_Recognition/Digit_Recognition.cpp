// Digit_Recognition.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "float.h"
#include "time.h"
#include "Windows.h"

#define K 32					//Size of the codebook
#define DELTA 0.00001			//Variable to store the value of DELTA
#define EPSILON 0.03			 //To know how to split clusters
#define UNIVERSE_SIZE 50000		//Taking 50000 as the size of the universe 
#define CLIP 5000				//Max value after normalizing
#define FS 320					//Frame Size
#define Q 12					//No. of cepstral coefficient
#define P 12					//No. of LPC
#define pie (22.0/7)
#define N 5						//no. of states in HMM Model
#define M 32					//Codebook Size
#define T_ 400					//Max possible no. of frames
#define TRAIN_SIZE 20			//Training Files for each utterance
#define TEST_SIZE 50			//Total Test Files if Train Size is 25

//HMM Model Variables
long double A[N + 1][N + 1],B[N + 1][M + 1], pi[N + 1], alpha[T_ + 1][N + 1], beta[T_ + 1][N + 1], gamma[T_ + 1][N + 1], delta[T_+1][N+1], xi[T_+1][N+1][N+1], A_bar[N + 1][N + 1],B_bar[N + 1][M + 1], pi_bar[N + 1];
int O[T_+1], q[T_+1], psi[T_+1][N+1], q_star[T_+1];
long double P_star=-1, P_star_dash=-1;

//Store 1 file values
int samples[50000];
//No. of frames in file
int T=160;
//Index to represent the start frame of the speech
int start_frame;
//Index to represent the end frame of the speech
int end_frame;

//Durbin's Algorithm variables
long double R[P+1];
long double a[P+1];
//Cepstral Coefficients
long double C[Q+1];
//Store codebook
long double reference[M+1][Q+1];
//Tokhura Weights
long double tokhuraWeight[Q+1]={0.0, 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
//Store energry per frame
long double energy[T_]={0};
//Universe vector
long double X[UNIVERSE_SIZE][Q];
//Universe Vector size
int LBG_M=0;
//Codebook
long double codebook[K][Q];
//Store mapping of universe with cluster
int cluster[UNIVERSE_SIZE];

/**************************************************************************** CREATING UNIVERSE SET **********************************************/

//Normalizing the data 
void normalization(char file[100]){
	FILE* fp=fopen(file,"r");
	if(fp==NULL)
	{
		printf("Error in Opening File!\n");
		return;
	}
	int amp=0,avg=0;
	int i=0;
	int n=0;
	int min_amp=INT_MAX;
	int max_amp=INT_MIN;
	while(!feof(fp))
	{
		fscanf(fp,"%d",&amp);
		avg+=amp;
		min_amp=(amp<min_amp)?amp:min_amp;
		max_amp=(amp>max_amp)?amp:max_amp;
		n++;
	}
	avg/=n;
	T=(n-FS)/80 + 1;
	if(T>T_) T=T_;
	min_amp-=avg;
	max_amp-=avg;
	fseek(fp,0,SEEK_SET);
	while(!feof(fp))
	{
		fscanf(fp,"%d",&amp);
		if(min_amp==max_amp)
		{
			amp=0;
		}
		else
		{
			amp-=avg;
			amp=(amp*CLIP)/((max_amp>min_amp)?max_amp:(-1)*min_amp);
			samples[i++]=amp;
		}
	}
	fclose(fp);
}


//Finding value of ai's for each frame using Durbin's Algo
void durbinAlgo()
{
	long double E=R[0];
	long double alpha[13][13];
	for(int i=1;i<=P;i++){
		double k;
		long double numerator=R[i];
		long double alphaR=0.0;
		for(int j=1;j<=(i-1);j++){
			alphaR+=alpha[j][i-1]*R[i-j];
		}
		numerator-=alphaR;
		k=numerator/E;
		alpha[i][i]=k;
		for(int j=1;j<=(i-1);j++){
			alpha[j][i]=alpha[j][i-1]-(k*alpha[i-j][i-1]);
			if(i==P){
				a[j]=alpha[j][i];
			}
		}
		E=(1-k*k)*E;
		if(i==P){
			a[i]=alpha[i][i];
		}
	}
}


void autoCorrelation(int fno){
	long double s[FS];
	int fsi=fno*80;     //frame starting index
	for(int i=0;i<FS;i++)
	{
		long double wn=0.54-0.46*cos((2*(22.0/7.0)*i)/(FS-1));
		s[i]=wn*samples[i+fsi];
	}
	for(int i=0;i<=P;i++)
	{
		long double sum=0.0;
		for(int y=0;y<=FS-1-i;y++)
		{
			sum+=((s[y])*(s[y+i]));
		}
		R[i]=sum;
	}
 
	durbinAlgo();
}


void cepstralTransformation(){
	C[0]=2.0*(log(R[0])/log(2.0));
	for(int m=1;m<=P;m++){
		C[m]=a[m];
		for(int k=1;k<m;k++){
			C[m]+=((k*C[k]*a[m-k])/m);
		}
	}
}

//Using raised Sine window on Cepstral Coefficients
void raisedSineWindow(){
	for(int m=1;m<=P;m++){
		long double wm=(1+(Q/2)*sin(pie*m/Q));
		C[m]*=wm;
	}
}

//Finding ci's for each frame of the file and storing it in universe.csv file
void universe_util(FILE* fp, char file[]){
	normalization(file);
	int m=0;
	int nf=T;
	for(int f=0;f<nf;f++){
		autoCorrelation(f);
		cepstralTransformation();
		raisedSineWindow();
		for(int i=1;i<=Q;i++){
			fprintf(fp,"%Lf,",C[i]);
		}
		fprintf(fp,"\n");
	}
}


//To create the universe from the 300 recording samples
void createuniverse()
{
	FILE* f1;
	f1=fopen("234101012_universe.csv","w");  
	for(int d=0;d<=9;d++){
		for(int u=1;u<=TRAIN_SIZE;u++){
			char fname[60];
			sprintf(fname,"234101012_dataset/English/txt/234101012_E_%d_%02d.txt",d,u);
			universe_util(f1,fname);
		}
	}
}


/********************************************************* UNIVERSE SET CREATION COMPLETED *************************************************************/


/******************************************************** USING THE UNIVERSE SET AND LBG TO CREATE CODEBOOK **********************************************/

void load_universe(char file[100]){
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	
	int i=0;
	long double c;
	printf("hi");
	while(!feof(fp)){
		//printf("inside loop\n");
		fscanf(fp,"%Lf,",&c);
		//printf("got here %Lf\n",c);
		
		if(c==0.000 || c==-1.000)
		break;
		X[LBG_M][i]=c;
		i=(i+1)%12;
		if(i==0) LBG_M++;
	}
	fclose(fp);
}

void store_codebook(char file[100],int k){
	FILE* fp=fopen(file,"w");
	if(fp==NULL){
		printf("Error opening file!\n");
		return;
	}
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			fprintf(fp,"%Lf,",codebook[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


void print_codebook(int k){
	printf("Codebook of size %d:\n",k);
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			printf("%Lf\t",codebook[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void initialize_with_centroid(){
	long double centroid[12]={0.0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[j]+=X[i][j];
		}
	}
	for(int i=0;i<12;i++){
		centroid[i]/=LBG_M;
		codebook[0][i]=centroid[i];
	}
	print_codebook(1);
}


long double calculate_distance(long double x[12], long double y[12]){
	long double distance=0.0;
	for(int i=0;i<12;i++){
		distance+=(tokhuraWeight[i+1]*(x[i]-y[i])*(x[i]-y[i]));
	}
	return distance;
}


void nearest_neighbour(int k){
	for(int i=0;i<LBG_M;i++){
		long double nn=DBL_MAX;
		int cluster_index;
		for(int j=0;j<k;j++){
			long double dxy=calculate_distance(X[i],codebook[j]);
			if(dxy<=nn){
				cluster_index=j;
				nn=dxy;
			}
		}
		cluster[i]=cluster_index;
	}
}


void codevector_update(int k){
	long double centroid[K][12]={0.0};
	int n[K]={0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[cluster[i]][j]+=X[i][j];
		}
		n[cluster[i]]++;
	}
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			codebook[i][j]=centroid[i][j]/n[i];
		}
	}
}


long double calculate_distortion(){
	long double distortion=0.0;
	for(int i=0;i<LBG_M;i++){
		distortion+=calculate_distance(X[i],codebook[cluster[i]]);
	}
	distortion/=LBG_M;
	return distortion;
}


void KMeans(int k)
{
	FILE* fp=fopen("distortion.txt","a");
	if(fp==NULL)
	{
		printf("Error pening file!\n");
		return;
	}
	int m=0;
	long double prev_D=DBL_MAX, cur_D=DBL_MAX;
	do
	{
		nearest_neighbour(k);
		m++;
		codevector_update(k);
		prev_D=cur_D;
		cur_D=calculate_distortion();
		printf("m=%d\t:\t",m);
		printf("Distortion:%Lf\n",cur_D);
		fprintf(fp,"%Lf\n",cur_D);
	}while((prev_D-cur_D)>DELTA);
	printf("Updated ");
	print_codebook(k);
	fclose(fp);
}

void LBG()
{
	printf("\nLBG Algorithm:\n");
	int k=1;
	initialize_with_centroid();
	while(k!=K){
		for(int i=0;i<k;i++){
			for(int j=0;j<12;j++){
				long double Yi=codebook[i][j];
				codebook[i][j]=Yi-EPSILON;
				codebook[i+k][j]=Yi+EPSILON;
			}
		}
		k=k*2;
		KMeans(k);
	}
}

void load_codebook(){
	FILE* fp;
	fp=fopen("234101012_codebook.csv","r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	for(int i=1;i<=M;i++){
		for(int j=1;j<=Q;j++){
			fscanf(fp,"%Lf,",&reference[i][j]);
		}
	}
	fclose(fp);
}

void createcodebook()
{
	load_universe("234101012_universe.csv");
	LBG();
	store_codebook("234101012_codebook.csv",K);
	load_codebook();
}

/*********************************************************************** CODEBOOK CREATED**********************************************************/

/*********************************************************************** TRAINING THE MODEL *******************************************************/

//Initialize every variable of HMM module to zero
void initialization()
{
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			A[i][j] = 0;
		}
		for (int j = 1; j <= M; j++)
		{
			B[i][O[j]] = 0;
		}
		pi[i] = 0;
	}
	for (int i = 1; i <= T; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			alpha[i][j] = 0;
			beta[i][j] = 0;
			gamma[i][j] = 0;
		}
	}
}

//Forward Procedure
void calculate_alpha()
{
	//Initialization
	for (int i = 1; i <= N; i++)
	{
		alpha[1][i] = pi[i] * B[i][O[1]];
	}
	//Induction
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			long double sum = 0;
			for (int i = 1; i <= N; i++)
			{
				sum += alpha[t][i] * A[i][j];
			}
			alpha[t + 1][j] = sum * B[j][O[t + 1]];
		}
	}
	FILE *fp=fopen("alpha.txt","w");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp,"%e\t", alpha[t][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//Check the score
long double calculate_score()
{
	long double probability = 0;
	for (int i = 1; i <= N; i++)
	{
		probability += alpha[T][i];
	}
	return probability;
}

//Calculate Beta
//Backward Procedure
void calculate_beta()
{
	//Initailization
	for (int i = 1; i <= N; i++)
	{
		beta[T][i] = 1;
	}
	//Induction
	for (int t = T - 1; t >= 1; t--)
	{
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
			}
		}
	}
	FILE *fp=fopen("beta.txt","w");
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp,"%e\t", beta[t][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//Predict most individually likely states using gamma
//One of the solution to problem 2 of HMM
void predict_state_sequence(){
	for (int t = 1; t <= T; t++)
	{
		long double max = 0;
		int index = 0;
		for (int j = 1; j <= N; j++)
		{
			if (gamma[t][j] > max)
			{
				max = gamma[t][j];
				index = j;
			}
		}
		q[t] = index;
	}
	FILE* fp=fopen("predicted_seq_gamma.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",q[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
}

//Calculate Gamma
void calculate_gamma()
{
	for (int t = 1; t <= T; t++)
	{
		long double sum = 0;
		for (int i = 1; i <= N; i++)
		{
			sum += alpha[t][i] * beta[t][i];
		}
		for (int i = 1; i <= N; i++)
		{
			gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
		}
	}
	FILE *fp=fopen("gamma.txt","w");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp,"%.16e\t", gamma[t][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	predict_state_sequence();
}

//Solution to Problem2 Of HMM
void viterbi_algo(){
	//Initialization
	for(int i=1;i<=N;i++){
		delta[1][i]=pi[i]*B[i][O[1]];
		psi[1][i]=0;
	}
	//Recursion
	for(int t=2;t<=T;t++){
		for(int j=1;j<=N;j++){
			long double max=DBL_MIN;
			int index=0;
			for(int i=1;i<=N;i++){
				if(delta[t-1][i]*A[i][j]>max){
					max=delta[t-1][i]*A[i][j];
					index=i;
				}
			}
			delta[t][j]=max*B[j][O[t]];
			psi[t][j]=index;
		}
	}
	//Termination
	P_star=DBL_MIN;
	for(int i=1;i<=N;i++){
		if(delta[T][i]>P_star){
			P_star=delta[T][i];
			q_star[T]=i;
		}
	}
	//State Sequence (Path) Backtracking
	for(int t=T-1;t>=1;t--){
		q_star[t]=psi[t+1][q_star[t+1]];
	}
	FILE* fp=fopen("predicted_seq_viterbi.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for(int t=1;t<=T;t++){
		fprintf(fp,"%4d\t",q_star[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
}

//Calculate XI
void calculate_xi(){
	for(int t=1;t<T;t++){
		long double denominator=0.0;
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				denominator+=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j]);
			}
		}
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				xi[t][i][j]=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j])/denominator;
			}
		}
	}
}

//Reestimation; Solution to problem3 of HMM
void re_estimation(){
	//calculate Pi_bar
	for(int i=1;i<=N;i++){
		pi_bar[i]=gamma[1][i];
	}
	//calculate aij_bar
	for(int i=1;i<=N;i++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int j=1;j<=N;j++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T-1;t++){
				numerator+=xi[t][i][j];
				denominator+=gamma[t][i];
			}
			A_bar[i][j]=(numerator/denominator);
			if(A_bar[i][j]>max_value){
				max_value=A_bar[i][j];
				mi=j;
			}
			adjust_sum+=A_bar[i][j];
		}
		A_bar[i][mi]+=(1-adjust_sum);
	}
	//calculate bjk_bar
	for(int j=1;j<=N;j++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int k=1;k<=M;k++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T;t++){		
					if(O[t]==k){
						numerator+=gamma[t][j];
					}
					denominator+=gamma[t][j];
			}
			B_bar[j][k]=(numerator/denominator);
			if(B_bar[j][k]>max_value){
				max_value=B_bar[j][k];
				mi=k;
			}
			if(B_bar[j][k]<1.00e-030){
				B_bar[j][k]=1.00e-030;
			}
			adjust_sum+=B_bar[j][k];
		}
		B_bar[j][mi]+=(1-adjust_sum);
	}
	
	//update Pi_bar
	for(int i=1;i<=N;i++){
		pi[i]=pi_bar[i];
	}
	//upadte aij_bar
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_bar[i][j];
		}
	}
	//update bjk_bar
	for(int j=1;j<=N;j++){
		for(int k=1;k<=M;k++){
			B[j][k]=B_bar[j][k];
		}
	}
}

//Set initial model for each didgit
void set_initial_model(){
	for(int d=0;d<=9;d++){
		char srcfnameA[40];
		_snprintf(srcfnameA,40,"initial/A_%d.txt",d);
		char srcfnameB[40];
		_snprintf(srcfnameB,40,"initial/B_%d.txt",d);
		char destfnameA[40];
		_snprintf(destfnameA,40,"initial_model/A_%d.txt",d);
		char destfnameB[40];
		_snprintf(destfnameB,40,"initial_model/B_%d.txt",d);
		char copyA[100];
		_snprintf(copyA,100,"copy /Y %s %s",srcfnameA,destfnameA);
		char copyB[100];
		_snprintf(copyB,100,"copy /Y %s %s",srcfnameB,destfnameB);
		system(copyA);
		system(copyB);
	}
	
}

//Store initial values of HMM model parameter into arrays
void initial_model(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"initial_model/A_%d.txt",d);
	fp = fopen(filenameA, "r");
	if (fp == NULL)
	{
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"initial_model/B_%d.txt",d);
	fp = fopen(filenameB, "r");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}

//Train HMM Model for given digit and given utterance
void train_model(int digit, int utterance){
	int m=0;
	do{
		calculate_alpha();
		calculate_beta();
		calculate_gamma();
		P_star_dash=P_star;
		viterbi_algo();
		calculate_xi();
		re_estimation();
		m++;
		printf("Digit:%02d\tIteration:%02d\t : \tP*=%e\n",digit,m,P_star);
	}while(m<60 && P_star > P_star_dash);
	
	//Store A in file
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101012_lambda/A_%d_%02d.txt",digit,utterance);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	//Store B in file
	char filenameB[40];
	_snprintf(filenameB,40,"234101012_lambda/B_%d_%02d.txt",digit,utterance);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//Calculate average model parameter for given digit
void calculate_avg_model_param(int d){
	long double A_sum[N+1][N+1]={0};
	long double B_sum[N+1][M+1]={0};
	long double temp;
	FILE* fp;
	for(int u=1;u<=25;u++){
		char filenameA[100];
		sprintf(filenameA,"234101012_lambda/A_%d_%02d.txt",d,u);
		fp=fopen(filenameA,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				A_sum[i][j]+=temp;
			}
		}
		fclose(fp);
		char filenameB[100];
		sprintf(filenameB,"234101012_lambda/B_%d_%02d.txt",d,u);
		fp=fopen(filenameB,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= M; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				B_sum[i][j]+=temp;
			}
		}
		fclose(fp);
	}
	FILE* avgfp;
	char fnameA[40];
	_snprintf(fnameA,40,"initial_model/A_%d.txt",d);
	avgfp=fopen(fnameA,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_sum[i][j]/25;
			fprintf(avgfp,"%e ", A[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
	char fnameB[40];
	_snprintf(fnameB,40,"initial_model/B_%d.txt",d);
	avgfp=fopen(fnameB,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=M;j++){
			B[i][j]=B_sum[i][j]/25;
			fprintf(avgfp,"%e ", B[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
}

//Store converged Model Parameter
void store_final_lambda(int digit){
	FILE *fp;
	char filenameA[40];
	sprintf(filenameA,"234101012_lambda/A_%d.txt",digit);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	char filenameB[40];
	_snprintf(filenameB,40,"234101012_lambda/B_%d.txt",digit);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//calculate energy of frame
void calculate_energy_of_frame(int frame_no){
	int sample_start_index=frame_no*80;
	energy[frame_no]=0;
	for(int i=0;i<FS;i++){
		energy[frame_no]+=samples[i+sample_start_index]*samples[i+sample_start_index];
		energy[frame_no]/=FS;
	}
}

//Calculate Max Energy of file
long double calculate_max_energy(){
	int nf=T;
	long double max_energy=DBL_MIN;
	for(int f=0;f<nf;f++){
		if(energy[f]>max_energy){
			max_energy=energy[f];
		}
	}
	return max_energy;
}

//calculate average energy of file
long double calculate_avg_energy(){
	int nf=T;
	long double avg_energy=0.0;
	for(int f=0;f<nf;f++){
		avg_energy+=energy[f];
	}
	return avg_energy/nf;
}


//mark starting and ending of speech activity
void mark_checkpoints(){
	int nf=T;
	//Calculate energy of each frame
	for(int f=0;f<nf;f++){
		calculate_energy_of_frame(f);
	}
	//Make 10% of average energy as threshold
	long double threshold_energy=calculate_avg_energy()/10;
	int isAboveThresholdStart=1;
	int isAboveThresholdEnd=1;
	start_frame=0;
	end_frame=nf-1;
	//Find start frame where speech activity starts
	for(int f=0;f<nf-5;f++){
		for(int i=0;i<5;i++){
			isAboveThresholdStart*=(energy[f+i]>threshold_energy);
		}
		if(isAboveThresholdStart){
			start_frame=((f-5) >0)?(f-5):(0);
			break;
		}
		isAboveThresholdStart=1;
	}
	//Find end frame where speech activity ends
	for(int f=nf-1;f>4;f--){
		for(int i=0;i<5;i++){
			isAboveThresholdEnd*=(energy[f-i]>threshold_energy);
		}
		if(isAboveThresholdEnd){
			end_frame=((f+5) < nf)?(f+5):(nf-1);
			break;
		}
		isAboveThresholdEnd=1;
	}
}

//calculate minimium Tokhura Distance
int minTokhuraDistance(long double testC[]){
	long double minD=DBL_MAX;
	int minDi=0;
	for(int i=1;i<=M;i++){
		long double distance=0.0;
		for(int j=1;j<=Q;j++){
			distance+=(tokhuraWeight[j]*(testC[j]-reference[i][j])*(testC[j]-reference[i][j]));
		}
		if(distance<minD){
			minD=distance;
			minDi=i;
		}
	}
	return minDi;
}


//Generate Observation Sequence
void generate_observation_sequence(char file[]){
	FILE* fp=fopen("o.txt","w");
	//normalize data
	normalization(file);
	int m=0;
	//mark starting and ending index
	mark_checkpoints();
	T=(end_frame-start_frame+1);
	int nf=T;
	//long double avg_energy=calculate_avg_energy();
	//repeat procedure for each frames
	for(int f=start_frame;f<=end_frame;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window "or" liftering
		raisedSineWindow();
		fprintf(fp,"%d ",minTokhuraDistance(C));
	}
	fprintf(fp,"\n");
	fclose(fp);
}

void train()
{
	//Take Initial A,B,PI 
	set_initial_model();
	for(int d=0;d<=9;d++){
		for(int t=1;t<=2;t++){
			for(int u=1;u<=TRAIN_SIZE;u++){
				char filename[60];
				sprintf(filename,"234101012_dataset/English/txt/234101012_E_%d_%02d.txt",d,u);
				generate_observation_sequence(filename);
				initial_model(d);
				train_model(d,u);
			}
			calculate_avg_model_param(d);
		}
		store_final_lambda(d);
	}
}

/************************************************************* TRAING PROCESS COMPLETED ************************************************************************/

/************************************************************** TESTING PROCESS STARTED **************************************************************************/

//Store model parameters of given digit in array for test input
void processTestFile(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"234101012_lambda/A_%d.txt",d);
	fp=fopen(filenameA,"r");
	if (fp == NULL){
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"234101012_lambda/B_%d.txt",d);
	fp=fopen(filenameB,"r");
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}


int recognize_digit(){
	int rec_digit=-1;
	long double max_prob=DBL_MIN;
	for(int d=0;d<=9;d++){
		processTestFile(d);
		calculate_alpha();
		long double prob=calculate_score();
		printf("P(O|lambda%d)=%e\n",d,prob);
		if(prob>max_prob){
			max_prob=prob;
			rec_digit=d;
		}
	}
	return rec_digit;
}

void test()
{
	double accuracy=0.0;
	for(int d=0;d<=9;d++){
		for(int u=21;u<=30;u++){
			char filename[60];
			sprintf(filename,"234101012_dataset/English/txt/234101012_E_%d_%02d.txt",d,u);
			generate_observation_sequence(filename);
			printf("Digit=%d\n",d);
			int rd=recognize_digit();
			printf("Recognized Digit:%d\n",rd);
			if(rd==d){
				accuracy+=1.0;
			}
		}
	}
	printf("Accuracy:%f\n",accuracy);
}

void live(){
	char s[]="YES";
    while (strcmp(s, "YES") == 0) {
        Sleep(2000);
        system("Recording_Module.exe 2 live_input/test.wav live_input/test.txt");
        generate_observation_sequence("live_input/test.txt");
        int rd = recognize_digit();
        if (rd == -1) {
            printf("UNABLE TO RECOGNIZE THE DIGIT PLEASE SPEAK AGAIN\n");
        } else {
            printf("Recognized Digit:%d\n", rd);
        }
        printf("ENTER YES TO SPEAK AGAIN OR NO TO STOP THE PROCESS\n");
        scanf("%3s", s);  //%3s to read at most 3 characters
        while (getchar() != '\n');  // Clear the input buffer
	}
}



int _tmain(int argc, _TCHAR* argv[])
{
	createuniverse();
	createcodebook();
	train();
	test();
	live();
	return 0;
}