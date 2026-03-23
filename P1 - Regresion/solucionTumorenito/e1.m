% Datos para el ejercicio
x = [6.1543, 7.9194, 9.2181, 7.3821, 1.7627, 4.0571, 9.3547, 9.1690, 4.1027, 8.9365, 15];
y = [21.0518, 23.0857, 31.0830, 27.3933, 5.9044, 15.8872, 32.5721, 26.3197, 11.4262, 29.9518, 2];
n = length(x);

% Con regresión 1D (y = a*x + b)
A = [sum(x.*x) sum(x); sum(x) n];    % Saco la matriz principal
b = [sum(x.*y); sum(y)];             % Saco la matriz "solución" como una matriz de 2x1
sol = A\b;                           % Genero los coeficientes (A\b == inv(A)*b)

plot(x, y, 'o'); hold on;           % Dibujo los puntos
xr = [0 10];                        % Los puntos de 'x' van de 0 a 10
yr = [sol(2) sol(1)*10+sol(2)];     % Los puntos de 'y' van de b a a*10+b
plot(xr, yr, '-r');                 % Dibujo mi recta

% Con pseudoinversa
B = [x' ones(n, 1)];                    % Creo la matriz con las características (en vertical)
coefs = pinv(B) * y';                   % Saco los coeficientes con la pseudoinversa (en vertical)
xr = [0 10];                            % Los puntos de 'x' van de 0 a 10
yr = [coefs(2) coefs(1)*10+coefs(2)];   % Los puntos de 'y' van de b a a*10+b
plot(xr, yr, '-g'); hold off            % Y dibujo la recta con 'x' e 'y estimada'