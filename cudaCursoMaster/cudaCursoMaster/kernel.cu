
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>

/*
------------------------102- Introducction to parallel programing -------------------------------------------------------------------------------------------------------------------------------------------------------

Context (instruccion por turnos)
	- Collection of data about process which allows processor to suspend or hold the execution of a process and restart the execution later.
	- Memory addresses
	- Program counter states

Thread (secuendia mas peque�a de instruccion programada
	- Process
		- threads (subprocess)

Parallel Process
	- Tipos
		- Paralelismo a nivel de tarea
			- los nucleos realizan tareas distintas con datos distintos o los mismos
		- Paralelismo a nivel de datos
			- los nucleos realizan la misma tarea con diferentes datos

Paralelismo vs concurrencia.
	- Concurrencia = realizacion de procesos en distintos tiempos(secuenciales) de milesimas de segundo que aparentan simultaneidad o paralelismo
	- Paralelismo = distintos nucleos realizan tareas al mismo tiempo

------------------------104- Install -------------------------------------------------------------------------------------------------------------------------------------------------------
Revisar compatibilidad en wikipedia en ingles.
GPGPU- 
windows + r (dxdiag) // visualiza las caracteristicas del PC 
windows + r (cmd) //escribir (nvcc --version) para saber la version de CUDA instalado del PC

*/

//------------------------105 - Basic steps of a CUDA program----------------------------------------------------
/*RESUMEN
* - initization of data from CPU
* - transfer data from CPU context to GPU context
* - Kernel launc with needed grid/block size
* - Transfer results back to CPU context from CPU context
* - Reclaim the memory from both CPU and GPU
* 
* - IMPORTANTE
*	-Grid - Grid is a collection of all the threads launch for a kernel ( coleccion de todos los hilos lanzados para un kernel)
*		- En el ejemplo hello CUDA world se tienen 20 subprocesos, los hilos en una cuadricula estan organizados
*		  en un grupo llamado bloques de hilos
*	-Block - subconjunto de hilos dentro de un GRID que se pueden representar como un cubo(3d) mas peque�o que a su vez esta subdividido en peque�os cubos que representan a los hilos o threads
	
	-GRID (cubo general en x, y z)
		-BLOCK (subcubo dentro de GRID que forma un subconjunto de hilos)
			-THREADS

	kernel_ name <<<
				number_of_blocks, // especifica cuantos bloques de hilos en la cuadracula en cada dimension
				thread_per_block  // especifica cuantos hilos en un bloque en cada dimension
								>>> (arguments) // TODO ESTO ES EN UNA DIMENSION

* - Para especificar cuadriculas y bloques multidimensionales
*	-dim3 variable_name (x,y,z) // se inicializa por defecto en 1
*		- dim3 variable_name(x,y,z) // puede acceder a cada valor de dimension 
*			- variable_name.x
* *			- variable_name.y
* *			- variable_name.z

*EJEMPLO UNIDIMENSIONAL
* - 8 bloques de hilos, donde cada bloque tiene 4 hilos en la dimension x
	-la dimension de nuestro bloque es de cuatro hilos en la dimension x y 1 hilo en las dimensiones Y y Z.
		
		-dim3 grid(8,1,1) // nos referimos a todos los hilos lanzados para un kernel como grid.
		-dim3 block(4,1,1)
		 _________________________		 _________________________		 _________________________		 _________________________		 _________________________		 _________________________		 _________________________		 _________________________
		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |		|  | |   | |   | |   | |  |
		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |		|  |1|	 |2|   |3|   |4|  |	
		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|		|__|_|___|_|___|_|___|_|__|

 8 bloques unidimensionales con 8 hilos unidimensionales

 - si no se especifican las dimensiones se inicializaran como 1
 
 -LIMITES DE TAMA�O DE BLOQUE
	- 1024 HILOS PARA DIMENSION X 
	- 1024 HILOS PARA DIMENSION Y
	-   64 HILOS PARA DIMENSION Z

	- x* y* x <= 1024	la multiplicacion del numero de subprocesos en cada 
	  dimension deber ser menor o igual a 1024

-LIMITES DE TAMA�O DE CUADRICULA
	- 65536 (1<<32-1)  BLOQUES PARA DIMENSION X
	- 65536 (2^32-1)  BLOQUES PARA DIMENSION Y,Z




*/


//// EJEMPLOS 
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
////kernel
///*
//* funcion asincrona ( el host puede continuar con las siguientes instrucciones) a
//* menos que se especifique que debe esperar (cudaDeviceSynchronize())
//* 
//* cudaDeviceReset();  reestablece el dispositivo
//*/
//__global__ void hello_cuda()
//{
//	printf("Hello CUDA world \n");
//	//cout << "hello CPU world" << endl; // esta instruccion no funciona dentro del kernel
//}
//
//
//int main() 
//{
//	//************************// EJEMPLO 1 ***
//	
//	//hello_cuda <<<1,1 >>>(); // kernel con parametros de lanzamiento
//	/*
//	* - El segundo parametro hace referencia al numero de subprocesos que se ejecutanran en el DEVICE
//	* 
//	*/
//	//hello_cuda << <1, 10 >> > (); // imprime 10 veces hello CUDA world
//
//	//cudaDeviceSynchronize(); // hace que el CPU o host espere en este punto, hasta que termine el proceso de DEVICE
//
//	//cudaDeviceReset(); //reestablece dispositivo
//
//	//cout << "hello CPU world" << endl;
//
//	//return 0;
//
//	//************************// EJEMPLO 2 "8 BLOQUES(8X1) CON 4 HILOS(4X1)" imprime 32 veces hello CUDA world***
//
//	//dim3 grid(8); // conjunto de 8 blocks en X y 1 en las dimensiones Y,Z.
//	//dim3 block(4); // bloque con tama�o 4 en X y 1 en Y,Z.
//	//
//	//// el primer parametro(grid) es el numero de bloques de hilos en cada dimension
//	//// el segundo parametro(block) es el numero de hilos en cada dimension del bloque
//
//	//hello_cuda << <grid, block >> > (); 
//
//	//cudaDeviceSynchronize(); // hace que el CPU o host espere en este punto, hasta que termine el proceso de DEVICE
//
//	//cudaDeviceReset(); //reestablece dispositivo
//
//	//cout << "hello CPU world" << endl;
//
//	//return 0;
//
//	//************************// EJEMPLO 3 "4 BLOQUES (2X2) CON 16 HILOS (8X2) " imprime 32 veces hello CUDA world***
//
//	
//	int nx; // variables dinamicas para ir modificando en tiempo de ejecucion
//	int ny;
//	nx = 16;
//	ny = 4;
//	
//	
//	dim3 block(8,2); // 16 hilos en cada bloque
//	dim3 grid(nx/block.x, ny/block.y); // 16/8=2 , 4/2=2  4 bloques en total
////		 _____________________________________________
////		|  | |   | |   | |   | |  |	|  | |   | |   | |
////		|  |1|	 |2|   |3|   |4|  |5|  |6|	 |7|   |8| == 4  BLOQUES (GRID) IGUALES A ESTE.
////		|  |1|	 |2|   |3|   |4|  |5|  |6|	 |7|   |8|
////		|__|_|___|_|___|_|___|_|__|_|__|_|___|_|___|_|
//	    
//	// el primer parametro(grid) es el numero de bloques de hilos en cada dimension
//	// el segundo parametro(block) es el numero de hilos en cada dimension del bloque
//
//	hello_cuda << <grid, block >> > ();
//
//	cudaDeviceSynchronize(); // hace que el CPU o host espere en este punto, hasta que termine el proceso de DEVICE
//
//	cudaDeviceReset(); //reestablece dispositivo
//
//	cout << "hello CPU world" << endl;
//
//	return 0;
//
//}

//------------------------106 - Organization of threads in a CUDA program 1----------------------------------------------------

//1D
//      	 ______________________		     ____________________
//  		|  |A|   |B|   |C|   |D|		|E|  |F|     |G|   |H|

//Threadlx.X|  |0|	 |1|   |2|   |3|		|0|  |1|	 |2|   |3| 
//Threadlx.Y|  |0|	 |0|   |0|   |0|		|0|  |0|	 |0|   |0|  2 bloques : ejemplo de identificacion de hilo
//Threadlx.Z|  |0|	 |0|   |0|   |0|		|0|  |0|	 |0|   |0|				C = 2,0,0

//  		|__|_|___|_|___|_|___|_|		|_|__|_|_____|_|___|_|

//2D
//      	 __0_____1____2____3___		    _0___1_______2___3__
//  		|  ||   |X|   ||   ||			|P|  ||     ||   ||

//  		|  ||   |Y|   ||   ||			||   ||     |Q|  ||
//      	 ______________________		     ____________________
//																					X Y P Q			R S T U											
//      	 ______________________		     ____________________	 Threadlcx.X	1 1 0 2			0 3 1 0
//  		|  |R|   ||   ||   ||			||  |T|     ||   ||		 Threadlcx.Y	0 1 0 1			0 1 0 1	

//  		|  ||    ||   ||   |S|			|U|  ||     ||   ||
//      	 ______________________		     ____________________

//	//************************// EJEMPLO 1 -> GRID 2X2 CON 8 HILOS CADA BLOQUE ***
//	//************************// EJEMPLO 1 -> IDENTIFICACION DE HILOS ***

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//__global__ void print_threadIds()
//{
//	printf("threadIdx.x : %d,  threadIdx.y : %d,  threadIdx.z : %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
//}
//
//int main()
//{
//	int nx, ny;
//	nx = 2;
//	ny = 2;
//
//	dim3 block(2, 2); // 8 subprocesos en la dimension X y 8 subprocesos en la dimension Y.
//	dim3 grid(nx / block.x, ny / block.y);  // grid de 2x2
//
//	print_threadIds << <grid, block >> > ();
//	cudaDeviceSynchronize(); // da la orden para que el host o int main espere a que termine el kernel o __global__.
//	cudaDeviceReset();
//	return 0;
//
//	/*
//	ORDEN  GRID 2X2 = 4 BLOCKS , 4 HILOS POR BLOCK = 16 HILOS 
//
//	PRIMER BLOCK 1
//	  _____
//	 |	   |
//	 | A B |
//	 | C D |
//	 |_____|
//	
//	*BLOCKS					1	|	  2   |	   3	|    4
//						A B C D   E F G H   I J K L   M N O P			
//		Threadidx.X		0 1 0 1	  0 1 0 1	0 1 0 1	  0 1 0 1
//		Threadidx.Y		0 0 1 1	  0 0 1 1	0 0 1 1	  0 0 1 1
//		Threadidx.Z		0 0 0 0	  0 0 0 0	0 0 0 0	  0 0 0 0
//
//	*/
//}

//------------------------107 - Organization of threads in a CUDA program 2----------------------------------------------------

/*
* En tiempo de ejecucion CUDA la variable blckldx inicializada de forma unica para cada hilo dependiendo de las coordenadas de la pertenencia

blockldx.X = coordenadas de cada hilo tomando como base cada block

//1D				   0					        1
//      	 ____________________		    __________________
//  		  |P|   ||   |Q|   ||			||  |R|   ||   |S|   
//     0  	 ____________________		    __________________						P Q R S			T U V X
//																	blockldx.X		0 0	1 1			0 0 1 1
//      	 ____________________		    __________________		blockldx.Y		0 0 0 0			1 1 1 1
//  		  |T|   ||   |U|   ||			|V|  ||   ||   |X|
//     1  	 ____________________		    __________________

//--------------------------------------------------------------------------
//					__________________X_______________
//2D				   0					        1
//      	 ____________________		    ___________________
//  		  ||   |X|   ||   ||			|P|  ||     ||   ||
//    |0
//    |		  ||   |Y|   ||   ||			||   ||     |Q|  ||
//    |   	 ______________________		     ____________________
//	 Y|																				X Y P Q			R S T U
//    |		______________________		     ____________________	 blockldx.X		0 0 1 1			0 0 1 1
//    |		  |R|   ||   ||   ||			||  |T|     ||   ||		 blockldx.Y		0 0 0 0			1 1 1 1
//    |1															 blockDim.X = 4
//  		  ||    ||   ||   |S|			|U|  ||     ||   ||		 blockDim.Y = 2
//      	 ______________________		     ____________________	 GridDim.X =  2
																	 GridDim.Y =  2

blockDim = es la dimension del bloque ej. blockDim.x=4 y blockDim.y = 2 da como resultado un bloque de 8 hilos
GridDim = es la dimension de la rejilla ej. gridDim.x = 2 y gridDim.y = 2 da como resultado 4 bloques de hilos

*/
//EJEMPLO

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//__global__ void print_details()
//{
//	printf("blockIdx.x : %d,  blockIdx.y : %d,  blockIdx.z : %d, blockDim.x : %d,  blockDim.y : %d, gridDim.x : %d,  gridDim.y : %d \n", blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
//}
//
//int main()
//{
//	int nx, ny;
//	nx = 4;
//	ny = 1;
//
//	dim3 block(1, 1); // 8 subprocesos en la dimension X y 8 subprocesos en la dimension Y.
//	dim3 grid(nx / block.x, ny / block.y);  // grid de 2x2
//
//	print_details << <grid, block >> > ();
//	cudaDeviceSynchronize(); // da la orden para que el host o int main espere a que termine el kernel o __global__.
//	cudaDeviceReset();
//	return 0;
//}


//------------------------108 - Ejercicio grid 3d y block 3d----------------------------------------------------

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//__global__ void print_details()
//{
//	printf("threadIdx.x : %d,  threadIdx.y : %d,  threadIdx.z : %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
//	printf("blockIdx.x : %d,  blockIdx.y : %d,  blockIdx.z : %d, blockDim.x : %d,  blockDim.y : %d,  blockDim.z : %d, gridDim.x : %d,  gridDim.y : %d,  gridDim.z : %d \n", blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
//}
//
//int main()
//{
//	int nx, ny, nz;
//	nx = 4;
//	ny = 4;
//	nz = 4;
//
//	dim3 block(2, 2, 2); // 8 subprocesos en la dimension X y 8 subprocesos en la dimension Y.
//	dim3 grid(nx / block.x, ny / block.y, nz / block.z);  // grid de 2x2
//
//	print_details << <grid, block >> > ();
//	cudaDeviceSynchronize(); // da la orden para que el host o int main espere a que termine el kernel o __global__.
//	cudaDeviceReset();
//	return 0;
//}

//------------------------109 Unique index calculation using threadIdx blockId and blockDim--------------------
//************************************Ejemplo 1
////asignar valores de un array a cada hilo
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//__global__ void unique_idx_calc_threadIdx(int* input)
//{
//	int tid = threadIdx.x;
//	printf("threadIdx : %d, value : %d \n", tid, input[tid]);
//}
//
//int main()
//{
//	int array_size = 8;
//	int array_byte_size = sizeof(int) * array_size;
//	int h_data[] = { 23,9,4,53,65,12,1,33 };
//
//	for (int i = 0; i < array_size; i++)
//	{
//		cout << h_data[i] << " ";
//	}
//
//	cout << endl;
//	cout << endl;
//
//	int* d_data;
//	cudaMalloc((void**)&d_data, array_byte_size);
//	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
//
//	//dim3 block(8); //8 threads en un bloque
//	//dim3 grid(1);
//
//	dim3 block(4); // 8 threads en 2 bloques de 4 cada uno
//	dim3 grid(2);
//
//	unique_idx_calc_threadIdx << <grid, block >> > (d_data);
//	cudaDeviceReset();
//	return 0;
//
//
//}

//************************************//Ejemplo 2
//asignar valores de un array continuos a un grupo de blocks (grid 1D con 16 hilos en 4 bloques)
// gid = tid + offset
// gid = tid + blackldx.x * blockDim.x

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;



__global__ void unique_gid_calculation(int * input)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x; // numero de hilos que componen un bloque 
	int gid = tid + offset; //indice en el que empezara a asignar valores a cada bloque de hilos

	//ejemplo de 3 blocks
/*

//1D						   0					            1							      3
//			      	  _______________________		    ________________________		________________________
//  				  |23|   |9|   |4|   |53|			|65|   |12|   |1|   |33|		|65|   |12|   |1|   |33|
tid(threadIdx)=        0      1     2      3	          0     1      2      3			  0     1      2      3
blackIdx.x	  =		   0	  0     0	   0			  1     1      1      1			  2     2      2      2
blockDim.x    =        4	  4		4	   4			  4	    4	   4	  4			  4	    4	   4	  4
offset		  =		   0      0     0      0              4     4      4      4			  8     8      8      8
gid			  =		   0	  1     2      3			  4     5      6      7           8     9      10     11

*/

	printf("blockIdx.x : %d, threadIdx.x : %d, gid: %d, value : %d \n",
		blockIdx.x, tid, gid, input[gid]);
	 
}

int main()
{
	int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = { 23,9,4,53,65,12,1,33,22,1,1,3,5,2,1,3 };

	for (int i = 0; i < array_size; i++)
	{
		cout << h_data[i] << " ";
	}

	cout << endl;
	cout << endl;

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	//dim3 block(8); //8 threads en un bloque
	//dim3 grid(1);

	dim3 block(4); // 4 threads en 4 bloques 
	dim3 grid(4);

	unique_gid_calculation << <grid, block >> > (d_data);
	cudaDeviceReset();
	return 0;


}
