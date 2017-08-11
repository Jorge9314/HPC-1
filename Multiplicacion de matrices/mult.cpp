#include <bits/stdc++.h>
using namespace std;
#define dbg(x) cout << #x << ": " << x << endl

void print(int *M, int rows, int cols) {
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
  int out;
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

void receive(int *M, string file_name, int &row, int &col) {
  cout << "--------RECEIVE---------" << endl;
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
        M = (int*)malloc(row*col*sizeof(int));
      } else { // each file with its columns
        v = split(line);
        for (int j = 0; j < col; j++) {
          M[row_aux * col + j] = tonum(v[j]);
          cout << "M[" << row_aux * col + j << "] = " << M[row_aux * col + j] << endl;
        }
        row_aux++;
      }
      i++;
    }
    myfile.close();
  }

  cout << "---------------" << endl;

  // dbg(row);
  // dbg(col);
}

void mult(int* X, int filX, int colX, int* Y, int filY, int colY, int* Z){
	for(int i = 0; i < filX; i++){
		for(int j = 0; j< colY; j++){
			int suma = 0;
			for(int k = 0; k< filY; k++){
				suma = suma + X[i * colX + k] * Y[ k * colY + j];
			}
			Z[i * colY + j] = suma;
		}
	}
}

int main(void) {
  int rowsA, colsA, rowsB, colsB;
  int *A, *B, *C;

  receive(A, "in1.in", rowsA, colsA);
  dbg(A[0]);
  receive(B, "in2.in", rowsB, colsB);

  dbg(rowsA), dbg(colsA);
  print(A, rowsA, colsA);

  dbg(rowsB), dbg(colsB);
  print(B, rowsB, colsB);

  assert(colsA != rowsB); // must be equal

  C = (int*)malloc(rowA*colB*sizeof(int));

  delete A;
  delete B;
  return 0;
}
