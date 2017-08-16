#include <bits/stdc++.h>
using namespace std;
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

void mult(float* A, int rowsA, int colsA, float* B, int rowsB, int colsB, float* C){
  for(int i = 0; i < rowsA; i++){
    for(int j = 0; j< colsB; j++){
      int suma = 0;
      for(int k = 0; k < rowsB; k++){
        suma = suma + A[i * colsA + k] * B[ k * colsB + j];
      }
      C[i * colsB + j] = suma;
    }
  }
}

int main(void) {
  clock_t startCPU, endCPU;
  int rowsA, colsA, rowsB, colsB;
  float *A = receive("in1.in", rowsA, colsA);
  float *B = receive("in2.in", rowsB, colsB);

  dbg(rowsA), dbg(colsA);
  print(A, rowsA, colsA);

  dbg(rowsB), dbg(colsB);
  print(B, rowsB, colsB);

  assert(colsA == rowsB); // must be equal

  float *C = (float*)malloc(rowsA*colsB*sizeof(float));

  startCPU = clock();
  mult(A, rowsA, colsA, B, rowsB, colsB, C);
  endCPU = clock();

  print(C, rowsA, colsB);

  write(C, rowsA, colsB);

  delete A;
  delete B;
  delete C;

  double time_CPU = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;
	cout << "time was: " << time_CPU << endl;

  return 0;
}
