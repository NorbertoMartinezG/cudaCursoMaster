
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>

/*
------------------------02- Introducction to parallel programing -------------------------------------------------------------------------------------------------------------------------------------------------------

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

------------------------04- Install -------------------------------------------------------------------------------------------------------------------------------------------------------
Revisar compatibilidad en wikipedia en ingles.
GPGPU- 
windows + r (dxdiag) // visualiza las caracteristicas del PC 
windows + r (cmd) //escribir (nvcc --version) para saber la version de CUDA instalado del PC

*/

//------------------------005 - Basic steps of a CUDA program------------------
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


// EJEMPLOS 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

//kernel
/*
* funcion asincrona ( el host puede continuar con las siguientes instrucciones) a
* menos que se especifique que debe esperar (cudaDeviceSynchronize())
* 
* cudaDeviceReset();  reestablece el dispositivo
*/
__global__ void hello_cuda()
{
	printf("Hello CUDA world \n");
	//cout << "hello CPU world" << endl; // esta instruccion no funciona dentro del kernel
}


int main() 
{
	//************************// EJEMPLO 1 ***
	
	//hello_cuda <<<1,1 >>>(); // kernel con parametros de lanzamiento
	/*
	* - El segundo parametro hace referencia al numero de subprocesos que se ejecutanran en el DEVICE
	* 
	*/
	//hello_cuda << <1, 10 >> > (); // imprime 10 veces hello CUDA world

	//cudaDeviceSynchronize(); // hace que el CPU o host espere en este punto, hasta que termine el proceso de DEVICE

	//cudaDeviceReset(); //reestablece dispositivo

	//cout << "hello CPU world" << endl;

	//return 0;

	//************************// EJEMPLO 2 "8 BLOQUES(8X1) CON 4 HILOS(4X1)" imprime 32 veces hello CUDA world***

	//dim3 grid(8); // conjunto de 8 blocks en X y 1 en las dimensiones Y,Z.
	//dim3 block(4); // bloque con tamaño 4 en X y 1 en Y,Z.
	//
	//// el primer parametro(grid) es el numero de bloques de hilos en cada dimension
	//// el segundo parametro(block) es el numero de hilos en cada dimension del bloque

	//hello_cuda << <grid, block >> > (); 

	//cudaDeviceSynchronize(); // hace que el CPU o host espere en este punto, hasta que termine el proceso de DEVICE

	//cudaDeviceReset(); //reestablece dispositivo

	//cout << "hello CPU world" << endl;

	//return 0;

	//************************// EJEMPLO 3 "4 BLOQUES (2X2) CON 16 HILOS (8X2) " imprime 32 veces hello CUDA world***

	
	int nx; // variables dinamicas para ir modificando en tiempo de ejecucion
	int ny;
	nx = 16;
	ny = 4;
	
	
	dim3 block(8,2); // 16 hilos en cada bloque
	dim3 grid(nx/block.x, ny/block.y); // 16/8=2 , 4/2=2  4 bloques en total
//		 _____________________________________________
//		|  | |   | |   | |   | |  |	|  | |   | |   | |
//		|  |1|	 |2|   |3|   |4|  |5|  |6|	 |7|   |8| == 4  BLOQUES (GRID) IGUALES A ESTE.
//		|  |1|	 |2|   |3|   |4|  |5|  |6|	 |7|   |8|
//		|__|_|___|_|___|_|___|_|__|_|__|_|___|_|___|_|
	    
	// el primer parametro(grid) es el numero de bloques de hilos en cada dimension
	// el segundo parametro(block) es el numero de hilos en cada dimension del bloque

	hello_cuda << <grid, block >> > ();

	cudaDeviceSynchronize(); // hace que el CPU o host espere en este punto, hasta que termine el proceso de DEVICE

	cudaDeviceReset(); //reestablece dispositivo

	cout << "hello CPU world" << endl;

	return 0;

}

