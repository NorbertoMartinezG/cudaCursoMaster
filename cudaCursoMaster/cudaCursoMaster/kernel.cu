
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

Thread (secuendia mas pequeña de instruccion programada
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
*	-Block - subconjunto de hilos dentro de un GRID que se pueden representar como un cubo(3d) mas pequeño que a su vez esta subdividido en pequeños cubos que representan a los hilos o threads
	
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
 
 -LIMITES DE TAMAÑO DE BLOQUE
	- 1024 HILOS PARA DIMENSION X 
	- 1024 HILOS PARA DIMENSION Y
	-   64 HILOS PARA DIMENSION Z

	- x* y* x <= 1024	la multiplicacion del numero de subprocesos en cada 
	  dimension deber ser menor o igual a 1024

-LIMITES DE TAMAÑO DE CUADRICULA
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
//	//dim3 block(4); // bloque con tamaño 4 en X y 1 en Y,Z.
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

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//
//
//__global__ void unique_gid_calculation(int * input)
//{
//	int tid = threadIdx.x;
//	int offset = blockIdx.x * blockDim.x; // numero de hilos que componen un bloque 
//	int gid = tid + offset; //indice en el que empezara a asignar valores a cada bloque de hilos
//
//	//ejemplo de 3 blocks
///*
//
////1D						   0					            1							      3
////			      	  _______________________		    ________________________		________________________
////  				      |23|   |9|   |4|   |53|			|65|   |12|   |1|   |33|		|65|   |12|   |1|   |33|
//tid(threadIdx)  =        0      1     2      3	          0     1      2      3			  0     1      2      3
//blockIdx.x	  =		   0	  0     0	   0			  1     1      1      1			  2     2      2      2
//blockDim.x      =        4	  4		4	   4			  4	    4	   4	  4			  4	    4	   4	  4
//offset		  =		   0      0     0      0              4     4      4      4			  8     8      8      8
//gid			  =		   0	  1     2      3			  4     5      6      7           8     9      10     11
//
//*/
//
//	printf("blockIdx.x : %d, threadIdx.x : %d, gid: %d, value : %d \n",
//		blockIdx.x, tid, gid, input[gid]);
//	 
//}
//
//int main()
//{
//	int array_size = 16;
//	int array_byte_size = sizeof(int) * array_size;
//	int h_data[] = { 23,9,4,53,65,12,1,33,22,1,1,3,5,2,1,3 };
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
//	dim3 block(4); // 4 threads en 4 bloques 
//	dim3 grid(4);
//
//	unique_gid_calculation << <grid, block >> > (d_data);
//	cudaDeviceReset();
//	return 0;
//
//
//}


//------------------------110  Unique index calculation for 2D grid 1--------------------
//------------------------110  calculo del indice global para cuadricula 2D 1 (GRID DE 2X2 CON 4x1 hilos) ------
/*
* 
* Formula para calcular el indice unico para identificar los hilos que estan en una segunda fila
* 
* Index = row offset + block offset + tid
* row offset = number of threads in one thread block row (blockldx.y)
* block offset = number of threads in thread block(blockldx.x)
* tid = threadldx.x
* 
* gid = gridDim.x * blockDim.x * blockldx.y + blockldx.x * blockDim.x + threadldx.x

*/

//asignar valores de un array continuos a un grupo de blocks (grid 2D con 16 hilos en 4 bloques de 4x1 hilos)

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//
//
//__global__ void unique_gid_calculation_2d(int * input)
//{
//	int tid = threadIdx.x;
//	int offset = blockIdx.x * blockDim.x; // numero de hilos que componen un bloque 
//
//	int row_offset = blockDim.x * gridDim.x * blockIdx.y;
//
//	int gid = tid + offset + row_offset; //indice en el que empezara a asignar valores a cada bloque de hilos

	//ejemplo de 3 blocks
/*
//2D (4 BLOQUES EN UN GRID DE 2X2
//								0					            1						
//			      	  _______________________		    ________________________		
//fila 1 de bloques   |23|   |9|    |4|   |53|			|22|   |1|   |1|   |3|		

//			      	  _______________________		    ________________________
//fila 2 de bloques	  |65|   |12|   |1|   |33|			|5|    |2|   |1|   |3|

//fila 1 de bloques   |23|   |9|    |4|   |53|			 |65|   |12|   |1|   |33|
tid(threadIdx.X)=      0      1     2      3	          0     1      2      3			 
blockIdx.x	  =		   0	  0     0	   0			  1     1      1      1			 
blockDim.x    =        4	  4		4	   4			  4	    4	   4	  4			 
offset		  =		   0      0     0      0              4     4      4      4			  
blockIdx.y    =		   0	  0		0	   0			  0		0	   0	  0
gridDim.x     =        2      2     2      2			  2		2	   2	  2
rowOffset	  =		   0	  0     0      0			  0     0      0      0          
gid			  =		   0	  1     2      3			  4     5      6      7          

//fila 2 de bloques	 |65|   |12|   |1|   |33|			 |5|    |2|   |1|    |3|
tid(threadIdx.X)=      0      1     2      3	          0     1      2      3
blockIdx.x	  =		   0	  0     0	   0			  1     1      1      1
blockDim.x    =        4	  4		4	   4			  4	    4	   4	  4
offset		  =		   0      0     0      0              4     4      4      4
blockIdx.y    =		   1	  1		1	   1			  1		1	   1	  1
gridDim.x     =        2      2     2      2			  2		2	   2	  2
rowOffset	  =		   8	  8     8      8			  8     8      8      8
gid			  =		   8	  9     10     11			 12    13     14     15
rowOffset = blockDim.x * gridDim.x * blockIdx.y;
gid = tid + offset + row_offset; //indice en el que empezara a asignar valores a cada bloque de hilos
*/

//	printf("blockIdx.x : %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d - input: %d \n",
//		blockIdx.x, blockIdx.y, tid, gid, input[gid]);
//	 
//}
//
//int main()
//{
//	int array_size = 16;
//	int array_byte_size = sizeof(int) * array_size;
//	int h_data[] = { 23,9,4,53,65,12,1,33,22,1,1,3,5,2,1,3 };
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
//	dim3 block(4); // 4 threads en 4 bloques 
//	dim3 grid(2,2);
//
//	unique_gid_calculation_2d << <grid, block >> > (d_data);
//	cudaDeviceReset();
//	return 0;
//
//
//}


//-----------------111  Unique index calculation for 2D grid 2--------------------
//-----------------111  calculo del indice global para cuadricula 2D  (GRID DE 2X2 CON 2x2 hilos) -----
/*
*
* Formula para calcular el indice unico para identificar los hilos que estan en una segunda fila
*
* Index = row offset + block offset + tid
* row offset = number of threads in one thread block row (blockldx.y)
* block offset = number of threads in thread block(blockldx.x)
* tid = threadldx.x
*
* gid = gridDim.x * blockDim.x * blockldx.y + blockldx.x * blockDim.x + threadldx.x

*/

//asignar valores de un array continuos a un grupo de blocks (grid 2D con 16 hilos en 4 bloques de 2x2 hilos)

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//
//
//__global__ void unique_gid_calculation_2d_2d(int * input)
//{
//	int tid = blockDim.x * threadIdx.y + threadIdx.x;
//	
//	int num_threads_in_a_block = blockDim.x * blockDim.y;
//	int block_offset = blockIdx.x * num_threads_in_a_block;
//
//	int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
//	int row_offset = num_threads_in_a_row * blockIdx.y;
//
//	int gid = tid + block_offset + row_offset; //indice en el que empezara a asignar valores a cada bloque de hilos

	//ejemplo de 3 blocks
/*
//2D (4 BLOQUES EN UN GRID DE 2X2
//								   0					            1
//bloques 1 y 2    			___________					    __________
//fila 1					|23|   |9|    					|22|   |1|   
//fila 2					 |4|   |53|						|1|    |3|

//bloques 3 y 4				____________						__________
//fila 1					|65|   |12|   					|5|    |2|   
//fila 2					|1|    |33|						|1|    |3|

tid = blockDim.x * threadIdx.y + threadIdx.x;
num_threads_in_a_block = blockDim.x * blockDim.y;
block_offset = blockIdx.x * num_threads_in_a_block;
num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
row_offset = num_threads_in_a_row * blockIdx.y;
gid = tid + block_offset + row_offset;

//fila 1 de bloques			|23|   |9|    |4|   |53|		 |22|   |1|   |1|    |3|
blockDim.x    =				 2		2	   2     2			  2		2	   2      2
treadsIdx.x   =				 0		1	   0     1			  0	    1	   0	  1
treadsIdx.y   =				 0		0	   1     1			  0	    0	   1	  1
tid			  =				 0      1      2     3	          0     1      2      3
blockDim.y    =				 2		2	   2     2			  2		2	   2      2
num_threads_in_a_block =	 4      4      4     4	          4     4      4      4
blockIdx.x	  =				 0	    0      0     0			  1     1      1      1
block_offset  =				 0	    0      0     0			  4     4      4      4
gridDim.x     =				 2      2      2     2			  2		2	   2	  2
num_threads_in_a_row   =	 8      8      8     8			  8     8      8      8
blockIdx.y	  =				 0	    0      0     0			  0     0      0      0
rowOffset	  =				 0		0      0     0			  0     0      0      0
gid			  =				 0		1      2     3			  4     5      6      7

tid = blockDim.x * threadIdx.y + threadIdx.x;
num_threads_in_a_block = blockDim.x * blockDim.y;
block_offset = blockIdx.x * num_threads_in_a_block;
num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
row_offset = num_threads_in_a_row * blockIdx.y;
gid = tid + block_offset + row_offset;

//fila 2 de bloques			|65|   |12|   |1|   |33|		 |5|    |2|   |1|    |3|
blockDim.x    =				 2		2	   2     2			  2		2	   2      2
treadsIdx.x   =				 0		1	   0     1			  0	    1	   0	  1
treadsIdx.y   =				 0		0	   1     1			  0	    0	   1	  1
tid			  =				 0      1      2     3	          0     1      2      3
blockDim.y    =				 2		2	   2     2			  2		2	   2      2
num_threads_in_a_block =	 4      4      4     4	          4     4      4      4
blockIdx.x	  =				 0	    0      0     0			  1     1      1      1
block_offset  =				 0	    0      0     0			  4     4      4      4
gridDim.x     =				 2      2      2     2			  2		2	   2	  2
num_threads_in_a_row   =	 8      8      8     8			  8     8      8      8
blockIdx.y	  =				 1	    1      1     1			  1	    1      1      1
rowOffset	  =				 8      8      8     8			  8     8      8      8
gid			  =				 8		9      10    11			  12    13     14     15

*/

//	printf("blockIdx.x : %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d - input: %d \n",
//		blockIdx.x, blockIdx.y, tid, gid, input[gid]);
//	 
//}
//
//int main()
//{
//	int array_size = 16;
//	int array_byte_size = sizeof(int) * array_size;
//	int h_data[] = { 23,9,4,53,65,12,1,33,22,1,1,3,5,2,1,3 };
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
//	dim3 block(2,2); // 4 threads en cada block (2x2)
//	dim3 grid(2,2);  // 4 blocks (2x2)
//
//	unique_gid_calculation_2d_2d << <grid, block >> > (d_data);
//	cudaDeviceReset();
//	return 0;
//
//
//}

//-----------------112	CUDA MEMORY TRANSFER --------------------------------------------------

/*
-Two devices
	-HOST ( cpu- memory) - CPU - CACHES AND DRAM
	-DEVICE  ( gpu - internal gpu memory)- SM (stream multiprocess) - CACHES AND DRAM

- Para transferir memoria entre el host y el dispositivo
	cudaMemCpy(
		destination ptr, source ptr,
				size in byte, direction)
				* ptr = puntero 
				*destination ptr = hostToDevice o DeviceToHost o HostToHost (cudamemcpyhtod, cudamemcpydtoh, cudamemcpydtod)
		
*/

//Ejemplo 1. pasar datos a memoria del device en un solo bloque de hilos


//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <stdlib.h>
//#include <time.h>
//using namespace std;
//
//__global__ void mem_trs_test(int* input) // kernel que toma como un puntero a una matriz de enteros
//{
//	//cuadricula 1D con 2 bloques de hilos
//	int gid = blockIdx.x * blockDim.x + threadIdx.x; //indice global para acceder a elementos de la matriz
//	printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
//}
//
//
//
//int main()
//{
//	int size = 128; // tamaño de la matriz
//	int byte_size = size * sizeof(int);// cantidad de bytes que necesitamos para asignar a esta matriz 
//	int* h_input;	//asignar memoria del Host (la h_ es para indicar que es una variable del lenguaje principal)
//
//	//asignacion de memoria usando funcion malloc.
//	h_input = (int*)malloc(byte_size); // asinacion de bytes necesarios
//
//	//inicializacion aleatoria de la matriz con secuencia aleatoria de numeros
//	time_t t;
//	srand((unsigned)time(&t));
//	for (int i = 0; i < size; i++)
//	{
//		h_input[i] = (int)(rand() & 0xff);//valor aleatoria entre 0 y 255
//	}
//
//	int* d_input; // se utiliza d_ para indicar que es una variable de dispositivo
//
//	//asignacion de memoria en el dispositivo(gpu)
//	/*
//		C		   CUDA
//	malloc		cudaMalloc -- asignar memoria
//	memset		cudaMemset -- establece valores para una ubicacion de memoria dada
//	free		cudaFree   -- recupera la ubicacion de memoria especificada
//	
//	*/
//
//	// ** = puntero doble o puntero a un puntero
//	// &d_input = especifica tamaño de la memoria
//	cudaMalloc((void**)&d_input,byte_size); 
//
//	cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);// tranferir la matriz inicializada en el host al dispositivo
//	// h_input = puntero de origen
//	// d_input = puntero de destino en el device
//
//	//parametros de lanzamiento
//	dim3 block(64); // TODO: POR LO GENERAL SE MANTIENE EL TAMAÑO EN MULTIPLOS DE 32
//	dim3 grid(2);
//
//	mem_trs_test << <grid, block >> > (d_input);
//	cudaDeviceSynchronize();// hace que la ejecucion espere en este punto
//
//	cudaFree(d_input); // recuperar memoria 
//	free(h_input); // recuperar memoria 
//
//	cudaDeviceReset();
//	return 0;
//
//}

//***********************************************

//Ejemplo 2. pasar datos a memoria del device en varios bloques de hilos


//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <stdlib.h>
//#include <time.h>
//using namespace std;
//
//__global__ void mem_trs_test2(int* input, int size) // kernel que toma como un puntero a una matriz de enteros
//{ //int size = tamaño matriz
//	
//	//cuadricula 1D con 2 bloques de hilos
//	int gid = blockIdx.x * blockDim.x + threadIdx.x; //indice global para acceder a elementos de la matriz
//	
//													 
//	// CON ESTA VERIFICACION SOLO SE UTILIZAN LOS HILOS QUE MANEJARAN DATOS DADO EL INPUT												 
//	/*if (gid < size)
//	{
//		printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
//	}*/
//
//	// SIN LA VERIFICACION SE ACCEDE A LOS HILOS DE TODO EL GRID AUN CUANDO NO MANEJEN DATOS
//	printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
//}
//
//
//
//int main()
//{
//	int size = 150; // tamaño de la matriz
//	int byte_size = size * sizeof(int);// cantidad de bytes que necesitamos para asignar a esta matriz 
//	int* h_input;	//asignar memoria del Host (la h_ es para indicar que es una variable del lenguaje principal)
//
//	//asignacion de memoria usando funcion malloc.
//	h_input = (int*)malloc(byte_size); // asinacion de bytes necesarios
//
//	//inicializacion aleatoria de la matriz con secuencia aleatoria de numeros
//	time_t t;
//	srand((unsigned)time(&t));
//	for (int i = 0; i < size; i++)
//	{
//		h_input[i] = (int)(rand() & 0xff);//valor aleatoria entre 0 y 255
//	}
//
//	int* d_input; // se utiliza d_ para indicar que es una variable de dispositivo

	//asignacion de memoria en el dispositivo(gpu)
	/*
		C		   CUDA
	malloc		cudaMalloc -- asignar memoria
	memset		cudaMemset -- establece valores para una ubicacion de memoria dada
	free		cudaFree   -- recupera la ubicacion de memoria especificada

	*/

	// ** = puntero doble o puntero a un puntero
	// &d_input = especifica tamaño de la memoria
//	cudaMalloc((void**)&d_input, byte_size);
//
//	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);// tranferir la matriz inicializada en el host al dispositivo
//	// h_input = puntero de origen
//	// d_input = puntero de destino en el device
//
//	//parametros de lanzamiento
//	dim3 block(32); // TODO: POR LO GENERAL SE MANTIENE EL TAMAÑO EN MULTIPLOS DE 32
//	dim3 grid(5);
//
//	mem_trs_test2 << <grid, block >> > (d_input,size);
//	cudaDeviceSynchronize();// hace que la ejecucion espere en este punto
//
//	cudaFree(d_input); // recuperar memoria 
//	free(h_input); // recuperar memoria 
//
//	cudaDeviceReset();
//	return 0;
//
//}

//-----------------112	exercise GRID 3D --------------------------------------------------

//-----------------114 Sum array example with validity check --------------------------------------------------
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
////#include "cuda_common.cuh"
//
//#include <stdio.h>
//#include "common.h" // incluye metodo para comparar matrices
//
//// for random initialize
//#include <stdlib.h>
//#include <time.h>
//
//// for memset
//#include <cstring>
//using namespace std;
//
//__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
//{
//	int gid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (gid < size) // verificar si el indice global esta dentro del tamaño de nuestra matriz
//	{
//		c[gid] = a[gid] + b[gid];
//	}
//}
//
//// funcion para verificar resultado de gpu
//void sum_array_cpu(int* a, int* b, int* c, int size)
//{
//	for (int i = 0; i < size; i++)
//	{
//		c[i] = a[i] + b[i];
//	}
//}
//
//int main()
//{
//	int size = 10000; // tamaño de la matriz
//	int block_size = 128; // tamaño del bloque en 128
//	int num_bytes = size * sizeof(int); // tamaño necesario en bytes
//
//	// punteros host
//	int* h_a, * h_b, * gpu_results;
//	
//	int* h_c; // para verificacion en cpu
//
//	//asignacion de memoria para cada puntero
//	h_a = (int*)malloc(num_bytes);
//	h_b = (int*)malloc(num_bytes);
//	gpu_results = (int*)malloc(num_bytes);
//	
//	h_c = (int*)malloc(num_bytes);// para verificacion en cpu
//
//	//inicializacion aleatoria de cada matriz
//	time_t t;
//	srand((unsigned)time(&t));
//	for (int i = 0; i < size; i++)
//	{
//		h_a[i] = (int)(rand() & 0xFF); // valor generado entre 0 y 255
//	}
//	for (int i = 0; i < size; i++)
//	{
//		h_b[i] = (int)(rand() & 0xFF);
//	}
//
//	sum_array_cpu(h_a, h_b, h_c, size);
//
//	memset(gpu_results, 0, num_bytes);
//
//	// punteros device
//	int* d_a, * d_b, * d_c;
//	cudaMalloc((int**)&d_a, num_bytes);
//	cudaMalloc((int**)&d_b, num_bytes);
//	cudaMalloc((int**)&d_c, num_bytes);
//
//	//tranferencia de matriz h_a y h_b
//	cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);
//
//	//launching the grid
//	dim3 block(block_size); //tamaño de bloque 128 en la dimension X
//	dim3 grid((size / block.x) + 1); // (10000 / 128) + 128 = GRID 1D de 79 block de 128 hilos cada uno
//
//	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(gpu_results, d_c, num_bytes, cudaMemcpyDeviceToHost); // puntero de origen d_c, puntero de destino gpu_results
//
//	// COMPARACION DE RESULTADOS CPU Y GPU
//	compare_arrays(gpu_results, h_c, size);
//	
//	cudaFree(d_c);
//	cudaFree(d_b);
//	cudaFree(d_a);
//	 
//	free(gpu_results);
//	free(h_b);
//	free(h_a);
//
//	cudaDeviceReset();
//	return 0;
//
//
//}

//-----------------116 Error handling --------------------------------------------------

/*Types error
*	-Compile time errors
*		-Errors language syntax.
* 
*	-Run time errors
*		-Errors happens while program is running
* 
*/

//ejemplo  con la suma anterior

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda_common.cuh"

#include <stdio.h>
#include "common.h" // incluye metodo para comparar matrices

// for random initialize
#include <stdlib.h>
#include <time.h>

// for memset
#include <cstring>

#include "cuda_common.cuh"

using namespace std;

__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size) // verificar si el indice global esta dentro del tamaño de nuestra matriz
	{
		c[gid] = a[gid] + b[gid];
	}
}

// funcion para verificar resultado de gpu
void sum_array_cpu(int* a, int* b, int* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	int size = 10000; // tamaño de la matriz
	int block_size = 128; // tamaño del bloque en 128
	int num_bytes = size * sizeof(int); // tamaño necesario en bytes

	//ERROR (comprobacion)
	cudaError error;

	// punteros host
	int* h_a, * h_b, * gpu_results;
	
	int* h_c; // para verificacion en cpu

	//asignacion de memoria para cada puntero
	h_a = (int*)malloc(num_bytes);
	h_b = (int*)malloc(num_bytes);
	gpu_results = (int*)malloc(num_bytes);
	
	h_c = (int*)malloc(num_bytes);// para verificacion en cpu

	//inicializacion aleatoria de cada matriz
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++)
	{
		h_a[i] = (int)(rand() & 0xFF); // valor generado entre 0 y 255
	}
	for (int i = 0; i < size; i++)
	{
		h_b[i] = (int)(rand() & 0xFF);
	}

	sum_array_cpu(h_a, h_b, h_c, size);

	memset(gpu_results, 0, num_bytes);

	// punteros device
	int* d_a, * d_b, * d_c;
	
	//---------------------------
	//ERROR FORMA MANUAL
	/*error = cudaMalloc((int**)&d_a, num_bytes);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Error : %s \n", cudaGetErrorString(error));
	}*/

	//ERROR UTILIZANDO cuda_common.cuh
	
	gpuErrchk(cudaMalloc((int**)&d_a, num_bytes));
	gpuErrchk(cudaMalloc((int**)&d_b, num_bytes));
	gpuErrchk(cudaMalloc((int**)&d_c, num_bytes));

	//-------------------------------
	//cudaMalloc((int**)&d_a, num_bytes);
	//cudaMalloc((int**)&d_b, num_bytes);
	//cudaMalloc((int**)&d_c, num_bytes);

	//tranferencia de matriz h_a y h_b
	cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);

	//launching the grid
	dim3 block(block_size); //tamaño de bloque 128 en la dimension X
	dim3 grid((size / block.x) + 1); // (10000 / 128) + 128 = GRID 1D de 79 block de 128 hilos cada uno

	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	cudaMemcpy(gpu_results, d_c, num_bytes, cudaMemcpyDeviceToHost); // puntero de origen d_c, puntero de destino gpu_results

	// COMPARACION DE RESULTADOS CPU Y GPU
	compare_arrays(gpu_results, h_c, size);
	
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	 
	free(gpu_results);
	free(h_b);
	free(h_a);

	cudaDeviceReset();
	return 0;


}