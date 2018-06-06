#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265

int main()
{
	int x=2, y;
	for(float i=0; i<=360; i+=5)
	{
		y = ceil(sin(i/180*PI)*20)/2;
		//gotoxy(x, 12-y);printf("o");
		//x++;
	}
	

	return 0;
}
