#include<iostream>
using namespace std;


int main(void){
	
	int numBlock=32;
	while(1024%numBlock!=0){
		numBlock=numBlock-1;
	}cout<<numBlock<<endl;

	return 0;
}
