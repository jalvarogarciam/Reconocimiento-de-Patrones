load IRIS.MAT

y = x(4,:)';    % Salida deseada

% Entrenamiento
A = [x(1:3,:)' ones(150,1)];
coefs = pinv(A)*y;

% Test con los datos disponibles
yestim = A*coefs;

% Dibujo del resultado
plot(y,'r');hold on;
plot(yestim,'b');hold off