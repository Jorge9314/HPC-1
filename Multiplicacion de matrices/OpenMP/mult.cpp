#include <bits/stdc++.h>
using namespace std;
#include <omp.h>
#define dbg(x) cout << #x << ": " << x << endl

void print(float *M, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cout << M[i * cols + j] << " " ;
    }
    cout << endl;
  }
  cout << endl;
}

int tonum(string &s) {
  stringstream ss;
  ss << s;
  float out;
  ss >> out;
  return out;
}

vector<string> split(string s) { // split a string by ','
  // dbg(s);
  istringstream ss(s);
  string token;
  vector<string> v;

  while(getline(ss, token, ',')) {
    v.push_back(token);
  }
  return v;
}

float* receive(string file_name, int &row, int &col) {
  // cout << "--------RECEIVE---------" << endl;
  float *M;
  string line;
  int i = 0, row_aux = 0;
  vector<string> v;
  ifstream myfile(file_name);
  if (myfile.is_open()) {
    while ( getline(myfile,line) ) {
      if (i == 0) {
        row = tonum(line);
      }
      else if (i == 1) {
        col = tonum(line);
        M = (float*)malloc(row*col*sizeof(float));
        //M = new int[row*col];
      } else { // each file with its columns
        v = split(line);
        for (int j = 0; j < col; j++) {
          M[row_aux * col + j] = tonum(v[j]);
          // cout << "M[" << row_aux * col + j << "] = " << M[row_aux * col + j] << endl;
        }
        row_aux++;
      }
      i++;
    }
    myfile.close();
  }

  // cout << "---------------" << endl;
  return M;

  // dbg(row);
  // dbg(col);
}

void write(float *M, int row, int col) {
  ofstream myFile;
  myFile.open("out.out");
  myFile << row << endl;
  myFile << col << endl;
  for (int i = 0; i < row; i++) {
    string line;
    for (int j = 0; j < col; j++) {
      ostringstream os;
      os << M[i * col + j]; // float to string
      if (j + 1 == col) { // if is the last
        line += os.str();
      } else {
        line += os.str() + ",";
      }
    }
    myFile << line << endl;
  }
}

void writeTimeResult(float time, int rowsA, int colsA, int rowsB, int colsB) {
  ofstream myFile;
  myFile.open("time.txt", ios::out | ios::app );
  myFile << time << " ";
  myFile << rowsA << " ";
  myFile << colsA << " ";
  myFile << rowsB << " ";
  myFile << colsB << endl;
}

void mult(float* A, int rowsA, int colsA, float* B, int rowsB, int colsB, float* C){
  int i, j, k, nthreads, tid;
  float suma;
  #pragma omp parallel shared(A,B,C,nthreads) private(tid)
  {
    nthreads = omp_get_num_threads();
    // printf("Number of Threads: %d\n",nthreads);

    #pragma omp for private(i,j,k,suma)
    for(i = 0; i < rowsA; i++){
      tid = omp_get_thread_num();
      // printf("Thread id: %d\n",tid);
      for(j = 0; j< colsB; j++){
        suma = 0.0;
        for(k = 0; k < rowsB; k++){
          suma = suma + A[i * colsA + k] * B[ k * colsB + j];
        }
        C[i * colsB + j] = suma;
      }
    }

  }
}


int main(int argc, char** argv) {
  if (argc =! 3) {
    printf("Must be called with the names of the out files\n");
    return 1;
  }

  time_t start,end;
  int rowsA, colsA, rowsB, colsB;
  string file_name1(argv[1]);
  string file_name2(argv[2]);
  float *A = receive(file_name1, rowsA, colsA);
  float *B = receive(file_name2, rowsB, colsB);

  dbg(rowsA), dbg(colsA);
  // print(A, rowsA, colsA);

  dbg(rowsB), dbg(colsB);
  // print(B, rowsB, colsB);

  assert(colsA == rowsB); // must be equal

  float *C = (float*)malloc(rowsA*colsB*sizeof(float));


  time (&start);
  mult(A, rowsA, colsA, B, rowsB, colsB, C);
  time (&end);


  // print(C, rowsA, colsB);

  write(C, rowsA, colsB);

  delete A;
  delete B;
  delete C;


  double dif = difftime (end,start);
  printf ("Elasped time is %.2lf seconds.", dif );

  writeTimeResult(dif, rowsA, colsA, rowsB, colsB);

  return 0;
}
