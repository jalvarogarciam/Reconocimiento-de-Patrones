%{
Ejercicio 1: Regresión lineal
Dados los siguientes datos:
    x 6.1543 7.9194 9.2181 7.3821 1.7627 4.0571 9.3547 9.1690 4.1027 8.9365
    y 21.0518 23.0857 31.0830 27.3933 5.9044 15.8872 32.5721 26.3197 11.4262 29.9518
Se pide:

a) Realizar y dibujar el ajuste de una recta.

b) Añadir el dato x = 15, y = 2, y volver a realizar el ajuste. ¿Qué sucede con la recta?
Resolver los apartados a) y b) de dos formas:

Utilizando la solución de regresión 1D (suma de x2, suma de x, etc.).
Utilizando la solución GLM para 1D (usando la pseudoinversa).
%}

clc, clear all, close all


%% Demo Regresion lineal 1D por minimos cuadrados: 
%         Ordinary Least Squares (OLS)

% Datos de ejemplo
x = [ 6.1543 7.9194 9.2181 7.3821 1.7627 4.0571 9.3547 9.1690 4.1027 8.9365 ];
y = [ 21.0518 23.0857 31.0830 27.3933 5.9044 15.8872 32.5721 26.3197 11.4262 29.9518 ];






%% Utilizando la solución de regresión 1D (suma de x2, suma de x, etc.).
% Damos solución al sistema de ecuaciones 
A=[
    sum(x.*x)   sum(x)
    sum(x)    length(x)
]; 

b=[sum(x.*y) sum(y)]';
sol1D=A\b;

f = @(x) sol1D(1)*x + sol1D(2); %Definimos la recta como una función f(x)



%% Utilizando la solución GLM para 1D (usando la pseudoinversa).

B = [x' ones(width(x), 1)]; % Construimos la matriz
solGLM = pinv(B) * y'; % Calculamos la solución usando pinv



g = @(x) solGLM(1)*x + solGLM(2); %Definimos la recta como una función g(x)



%% b) Añadir nuevo dato
% Añadir el nuevo dato
x2 = [x 15]; % Agregar el nuevo valor de x
y2 = [y 2];  % Agregar el nuevo valor de y

% Usamos una de las 2 sonluciones (son iguales)
C = [x2' ones(size(x2'))]; % Construimos la matriz
solGLM2 = pinv(C) * y2'; % Calculamos la solución usando pinv

h = @(x) solGLM2(1)*x + solGLM2(2); %Definimos la recta como una función g(x)





%% Dibujo de las rectas
a = 0; b = 16;
axis([min(x2) max(x2) min(y2) max(y2)]); %Definimos los ejes
plot (x,y,'o'); hold on; %Pintamos los puntos
plot (x2(end), y2(end), 'or')
plot ([a,b], [f(a), f(b)],'r'); %Pintamos la recta
plot ([a, b], [g(a), g(b)],'--g'); %Pintamos la recta
plot ([a, b], [h(a), h(b)],'y'); %Pintamos la recta