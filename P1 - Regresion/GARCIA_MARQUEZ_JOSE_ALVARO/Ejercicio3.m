%{
Ejercicio 3: Regresión aplicada a un problema real

El fichero COVID-19.xls incluye datos del Coronavirus en el mundo desde el 
22 de febrero de 2020 hasta el 12 de marzo de 2020 (1:00 a.m.). 

Queremos predecir la progression del coronavirus en España asumiendo
que la tendencia esperada es similar a la tendencia en China. 
La función getdata que devuelve el número de casos activos, confirmados, muertes
y recuperaciones por cada país. Los parámetros de dicha función son: 
    - Dataset, 
    - País 
    - Parámetro de normalización por densidad de población (147 habitantes por kilómetro cuadrado
    en China y 93 habitantes por kilómetro cuadrado en España). Para normalizar
    los datos chinos, el parámetro debe de ser 1/147.

1. Realiza un modelo de regression sobre los casos activos normalizados en
china usando el modelo 𝑓(𝑥) = 𝑎𝑥𝑒^(𝑏𝑥+𝑐x^2). 
Consejo: Para convertir un dato de tipo datetime a uno numérico para regresión, 
usar x=days(date-min(date(:)))+1; siendo “date” el vector de datetime devuelto 
por la función getdata.

2. Mostrad gráficamente los datos normalizados y realizar una proyeción a
30 días desde la última fecha. (Figura 1).

3. Una vez ajustado el modelo, mirar la fecha de inicio en España donde el
ajuste tiene error mínimo.

4. Dibujar los casos activos en España y el modelo de regresión. (Figura 2). 


5. ¿Cuándo serán los casos en España menos que 1?

6. Repetir los pasos 3, 4 y 5 para 'Korea, South' y 'Italy'. (Ver figuras 3 y 4
respectivamente). La normalización de los parámetros son 1/515 para
'Korea, South' y 1/200 para 'Italy'.

%}




%{
    Para ajustar una regresión usando el modelo de y = a*e^(bx+cx^2) usando
    un modelo Lineal GLM, haré un cambio de variable tomando logaritmos a
    ambos lados y reducir el modelo no lineal, a un modelo lineal. De esta
    forma:

                        ln(y) =  ln(a*x*e^(bx+cx^2))

                        ln(y) =  ln(a) + ln(x) + ln(e^(bx+cx^2))

                        ln(y) =  ln(a) + ln(x) + bx + cx^2
                        
                        ln(y/x) = ln(a) + ln(x) + bx + cx^2

    Llegando a esta expresión, con un cambio de variables, llegamos a un modelo lineal que podremos ajustar usando GLM.
%}


% =========================================================================
% APARTADO 1: Modelo de regresión para China.
% =========================================================================

% Cargamos los datos de la base de datos en memoria 
dataset = readtable('COVID-19.csv');

% Extraemos los datos de china
[activosChina, confirmadosChina, muertesChina, recuperadosChina, fechasChina] = ...      
                                           getdata(dataset, 'China', 1/147);

% Convertimos las fechas a días numéricos
diasChina = days(fechasChina-min(fechasChina(:))) + 1;

% Filtramos los ceros para que el logaritmo no dé infinito
idx = activosChina > 0;         % Nos quedamos con los índices donde hay casos
diasChina = diasChina(idx);         % Filtramos los días
activosChina = activosChina(idx);      % Filtramos los casos activos

% Calculamos los coeficientes para la regresión de china
AChina =[diasChina.^2, diasChina, ones(length(diasChina), 1)];
coefsChina = pinv(AChina) * (log(activosChina) - log(diasChina));
% Usamos log(y) - log(x) para nuestra y.


a = exp(coefsChina(3));
b = coefsChina(2);
c = coefsChina(1);

modeloChina = @(x) a .* x .* exp(b.*x + c.*x.^2);


% =========================================================================
% APARTADO 2: Mostrar datos gráficamente y realizar una proyección a 30
% días desde el último registrado.
% =========================================================================
proyeccion = 1:(max(diasChina) + 30); % Proyección de 30 días extra

figure(1);
plot(diasChina, activosChina, '-w'); hold on;
plot(proyeccion, modeloChina(proyeccion), 'r', 'LineWidth', 2);
title('Modelo COVID-19 en China (Proyección a 30 días)');
xlabel('Días'); ylabel('Casos Activos Normalizados');
legend('Datos Reales', 'Ajuste GLM');
axis([0 max(proyeccion) 0 max(activosChina)*1.1]);
hold off;







% =========================================================================
% APARTADO 3: Buscar el DÍA de inicio óptimo en España
% =========================================================================
%{
Debido a que en españa empezó más tarde que en china, al igual que en china
comenzamos con un día 0 en el que empezó la pandemia, tenemos que buscar en
españa ese mismo día 0 que hace que la curva se ajuste lo mejor posible a
los datos que tenemos de España.

Ésto lo hacemos iterando entre los días que tenemos registrados de España,
simulando cada día como si fuese nuestro día 0, y calculando el error para
cada iteración. Así, cuando hayamos simulado todos los días como el
inicial, podremos calcular cual fue el que presentó mínimo error de todos
estos.
%}

nombresPaises = {'Spain', 'Korea, South', 'Italy'};
densidadesPaises = [1/93, 1/515, 1/200];
numFiguras = [2, 3, 4];

for k = 1:length(nombresPaises)
    paisActual = nombresPaises{k};
    densidadActual = densidadesPaises(k);
    
    %% Extraemos los datos
    [activosP, confirmados, muertes, recuperados, fechasP] = ...      
                                               getdata(dataset, paisActual, densidadActual);

    % =====================================================================
    % APARTADO 3: Buscar el DÍA de inicio óptimo en España
    % =====================================================================
    diasP = days(fechasP - min(fechasP(:))) + 1; 
    
    
    % Calculamos cuántos días tenemos registrados en España
    num_dias = length(diasP);
    
    % Aquí registraremos el error mínimo registrado y el índice del día en el que se encuentra
    minError = [inf, 0]; 
    
    % 2. Bucle para probar cada día como si fuera el "Día 1"
    for i = 1:num_dias 
        
        % CORRECTO: Extraemos los CASOS reales desde el día 'i'
        casosRealesIter = activosP(i:end); 
        
        % Generamos las predicciones del modelo de China para la duración de esos datos
        t = 1:length(casosRealesIter); 
        predicciones = modeloChina(t);
        
        % Calculamos el MSE entre casos reales y casos predichos
        mse = mean((casosRealesIter(:) - predicciones(:)).^2);
        %{
        Aquí usamos la media porque si lo hiciéramos con el sumatorio, cuanto
        menos días mirásemos (i más alta) menor sería dicho sumatorio,
        portanto, obtendríamos falsos mínimos
        %}
        
        % Guardamos el error mínimo
        if mse < minError(1)
            minError = [mse, i];
        end
    end
    
    % 4. El día de inicio óptimo es la fecha correspondiente al índice i
    diaIni = diasP(minError(2)); 
    disp(['La fecha de inicio estimada en España es: ', num2str(diaIni)]);
    
    
    % =========================================================================
    % APARTADO 4: Dibujar los casos activos en españa y el modelo de regresión
    % =========================================================================
    %{
    Como ya hemos calculado el dia 0 correcto para España, podemos usar el
    modelo de China a partir de ese día para proyectar los resultados a
    posteriori.
    %}
    
    % Recuperamos el índice del día de inicio óptimo
    idxInicio = minError(2);
    
    % Generamos las predicciones del modelo de China a partir del día óptimo
    num_dias = (length(activosP) - idxInicio + 1) + 90;
    tPais = 1:num_dias; 
    % 60 para añadir una proyección para días en los que aún no tenemos datos
    
    % Calculamos las predicciones
    predicciones = modeloChina(tPais);
    
    % Preparamos el eje X para la gráfica (en días de calendario)
    % El modelo empieza en el día: diasEsp(idxInicio)
    ejeX_modelo = diasP(idxInicio) : (diasP(idxInicio) + num_dias - 1);
    
    figure(numFiguras(k));
    % Dibujamos los datos reales de España (todos)
    plot(diasP, activosP, '-w', 'MarkerFaceColor', 'w', 'MarkerSize', 4); 
    hold on;
    
    % Dibujamos el modelo desplazado
    plot(ejeX_modelo, predicciones, 'r', 'LineWidth', 2);

    title(['Modelo COVID-19 en ', paisActual]);
    xlabel('Días'); ylabel('Casos Activos Normalizados');
    legend('Datos Reales', 'Ajuste China');
    grid on;
    hold off;
    
    
    
    % =========================================================================
    % APARTADO 5: ¿Cuándo serán los casos en España menos que 1?
    % =========================================================================
    % Localizamos el pico en las predicciones que ya calculamos
    [~, picoIdx] = max(predicciones);
    
    % Buscamos en la parte derecha (desde el pico hasta el final)
    bajadaPais = predicciones(picoIdx:end);
    
    % Localizamos el primer índice donde los casos bajan de 1
    indicesBajos = find(bajadaPais < 1, 1);
    
    if ~isempty(indicesBajos)
        % Calculamos el día relativo al inicio del modelo
        % Restamos 1 porque el primer índice de 'bajada' es el día del pico
        diaRelativoModelo = picoIdx + indicesBajos - 1;
        
        % Calculamos la fecha real sumando al día de inicio de España
        fechaInicioReal = fechasP(minError(2));
        fechaFinPandemia = fechaInicioReal + diaRelativoModelo;
        
        disp(['El modelo indica que los casos serán < 1 el día ', num2str(diaRelativoModelo), ' tras el inicio.']);
        disp(['Fecha estimada: ', datestr(fechaFinPandemia)]);
    else
        disp('No se encontró el fin de la pandemia en el rango proyectado.');
    end

end