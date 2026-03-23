%{
Ejercicio 3: Técnica leave-one-out y random sampling

Repita el ejercicio 1, pero usando las técnicas siguientes en lugar de 
crossvalidation, a ver si funcionan mejor, igual o peor.

    a) Método LOO (Leave-One-Out). Para esta técnica, el código es prácticamente
    igual, ya que LOO es igual que CV de orden igual al número de datos.
    
    b) Random sampling. Para esta técnica usa un bucle que realice 1000 
    iteraciones de una validación simple al 75% (75% de los datos para 
    entrenar el modelo, y el 25% restante para estimar el error)
%}

clc, clear all, close all;


%% --- 1. Generación de datos
rand('seed', 0);
randn('seed', 0);
x = rand(1,100);
y = exp(x.^3 - x.^2 + 0.01*x + 2) + 0.04 * randn(size(x));


numModelos = 5;

%--------------------------------------------------------------------------
%%   Método LOO (Leave-One-Out).
%--------------------------------------------------------------------------




erroresLOO = zeros(1,numModelos);


% --- Generación y testeo de los modelos
for k = 1:length(x)

    % Separamos los datos de validación de los de test usando crossval
    [xtrn,xtst,ytrn,ytst] = crossval(x,y,length(x),k);

    for modeloIdx = 1:numModelos

        if modeloIdx <=3 % 3 primeros polinomios simples

            % Obtenemos el modelo usando polifit
            p = polyfit(xtrn, ytrn, modeloIdx);

            % Obtenemos los valores estimados usando los valores de test.
            yestim = polyval(p,xtst);

        elseif modeloIdx == 4 % y = a + bx + cx^2 + dx^3 + esin(x) + fcos(x)
            
            % Obtenemos el modelo usando GLM
            A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', sin(xtrn)', ones(length(xtrn),1)];
            coefs = pinv(A) * ytrn';

            % Obtenemos los valores estimados usando los valores de test.
            yestim = coefs(1)*xtst.^3 + coefs(2)*xtst.^2 + coefs(3)*xtst + coefs(4)*sin(xtst) + coefs(5);

        elseif modeloIdx == 5 %  y = a + bx + cx^2 + dx^3 + e*sin(x) + f*cos(x)
            
            % Obtenemos el modelo usando GLM
            A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', sin(xtrn)', cos(xtrn)', ones(length(xtrn),1)];
            coefs = pinv(A) * ytrn';

            % Obtenemos los valores estimados usando los valores de test.
            yestim = coefs(1)*xtst.^3 + coefs(2)*xtst.^2 + coefs(3)*xtst + coefs(4)*sin(xtst) + coefs(5)*cos(xtst) + coefs(6);
        end

        erroresLOO(modeloIdx) = erroresLOO(modeloIdx) + sumsqr(yestim-ytst);

    end
end

% Calculamos los errores medios
erroresLOO = erroresLOO / length(x);




%--------------------------------------------------------------------------
%%   Método Random Sampling.
%--------------------------------------------------------------------------

erroresRS = zeros(1,numModelos);

numIteraciones = 1000;

% --- Generación y testeo de los modelos
for k = 1:numIteraciones

    % Separamos los datos de entrenamiento de los de test 
    n = length(x);
    trnIdxs = randperm(n, round(0.75*n));
    tstIdxs = setdiff(1:n, trnIdxs);

    xtrn = x(trnIdxs); ytrn = y(trnIdxs);
    xtst = x(tstIdxs); ytst = y(tstIdxs);

    for modeloIdx = 1:numModelos

        if modeloIdx <=3 % 3 primeros polinomios simples

            % Obtenemos el modelo usando polifit
            p = polyfit(xtrn, ytrn, modeloIdx);

            % Obtenemos los valores estimados usando los valores de test.
            yestim = polyval(p,xtst);

        elseif modeloIdx == 4 % y = a + bx + cx^2 + dx^3 + esin(x) + fcos(x)
            
            % Obtenemos el modelo usando GLM
            A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', sin(xtrn)', ones(length(xtrn),1)];
            coefs = pinv(A) * ytrn';

            % Obtenemos los valores estimados usando los valores de test.
            yestim = coefs(1)*xtst.^3 + coefs(2)*xtst.^2 + coefs(3)*xtst + coefs(4)*sin(xtst) + coefs(5);

        elseif modeloIdx == 5 %  y = a + bx + cx^2 + dx^3 + e*sin(x) + f*cos(x)
            
            % Obtenemos el modelo usando GLM
            A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', sin(xtrn)', cos(xtrn)', ones(length(xtrn),1)];
            coefs = pinv(A) * ytrn';

            % Obtenemos los valores estimados usando los valores de test.
            yestim = coefs(1)*xtst.^3 + coefs(2)*xtst.^2 + coefs(3)*xtst + coefs(4)*sin(xtst) + coefs(5)*cos(xtst) + coefs(6);
        end

        erroresRS(modeloIdx) = erroresRS(modeloIdx) + sumsqr(yestim-ytst);

    end
end


% Calculamos los errores medios
erroresRS = erroresRS / numIteraciones;




%--------------------------------------------------------------------------
%%   Método cross validation.
%--------------------------------------------------------------------------


gradoCV = 10;
erroresCV = zeros(1,numModelos);


% --- Generación y testeo de los modelos
for k = 1:gradoCV

    % Separamos los datos de entrenamiento de los de test usando crossval
    [xtrn,xtst,ytrn,ytst] = crossval(x,y,gradoCV,k);

    for modeloIdx = 1:numModelos

        if modeloIdx <=3 % 3 primeros polinomios simples

            % Obtenemos el modelo usando polifit
            p = polyfit(xtrn, ytrn, modeloIdx);

            % Obtenemos los valores estimados usando los valores de test.
            yestim = polyval(p,xtst);

        elseif modeloIdx == 4 % y = a + bx + cx^2 + dx^3 + esin(x) + fcos(x)
            
            % Obtenemos el modelo usando GLM
            A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', sin(xtrn)', ones(length(xtrn),1)];
            coefs = pinv(A) * ytrn';

            % Obtenemos los valores estimados usando los valores de test.
            yestim = coefs(1)*xtst.^3 + coefs(2)*xtst.^2 + coefs(3)*xtst + coefs(4)*sin(xtst) + coefs(5);

        elseif modeloIdx == 5 %  y = a + bx + cx^2 + dx^3 + e*sin(x) + f*cos(x)
            
            % Obtenemos el modelo usando GLM
            A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', sin(xtrn)', cos(xtrn)', ones(length(xtrn),1)];
            coefs = pinv(A) * ytrn';

            % Obtenemos los valores estimados usando los valores de test.
            yestim = coefs(1)*xtst.^3 + coefs(2)*xtst.^2 + coefs(3)*xtst + coefs(4)*sin(xtst) + coefs(5)*cos(xtst) + coefs(6);
        end

        erroresCV(modeloIdx) = erroresCV(modeloIdx) + sumsqr(yestim-ytst);

    end
end

% Calculamos los errores medios
erroresCV = erroresCV / gradoCV;






%% --- Generación de la Tabla de Resultados ---
Metodos = {'CV10'; 'LOO'; 'Random Sampling'};
Modelos_Nombres = {'Mod1_Recta', 'Mod2_Parabola', 'Mod3_Grado3', 'Mod4_Seno', 'Mod5_SenoCos'};

% Creamos una matriz con todos los errores calculados
Resultados = [erroresCV; erroresLOO; erroresRS];

TablaFinal = array2table(Resultados, 'VariableNames', Modelos_Nombres, 'RowNames', Metodos);

disp(' ');
disp('==================== TABLA COMPARATIVA DE ERRORES ====================');
disp(TablaFinal);
disp('======================================================================');

% Buscamos el mejor modelo según cada método
[~, mejorCV] = min(erroresCV);
[~, mejorLOO] = min(erroresLOO);
[~, mejorRS] = min(erroresRS);

fprintf('\nEl mejor modelo según CV10 es: %d', mejorCV);
fprintf('\nEl mejor modelo según LOO es: %d', mejorLOO);
fprintf('\nEl mejor modelo según Random Sampling es: %d\n', mejorRS);