
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
02- Introducction to parallel programing

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

*/