# Preisanalyse mithilfe linearer Regression

**Beschreibung des Problems und Lösung**

Mein Auftraggeber für dieses Projekt sind meine Eltern. Sie stehen dabei vor zwei konkreten Fragen im Zusammenhang mit Immobilien: Einerseits
möchten sie den ungefähren Wert einer Immobilie einschätzen, die sie bereits besitzen. Andererseits wollen sie bei einem möglichen Kauf besser 
beurteilen können, ob der verlangte Preis für eine Immobilie tatsächlich gerechtfertigt ist oder eher über dem zu erwartenden Marktwert liegt.

In der Praxis ist dies oft schwierig, da Immobilienpreise von vielen Faktoren abhängen und häufig lediglich auf Einschätzungen basieren, die sich auf
Vergleichsobjekte, Erfahrungswerte oder die Bewertung von Experten stützen, was jedoch nicht immer objektiv oder einheitlich ist.

Die Lösung besteht darin, ein Machine-Learning-Modell zu entwickeln, das auf Basis historischer Immobiliendaten den Marktwert einer Immobilie
vorhersagen kann. Dafür wird ein überwacht lernendes Regressionsmodell verwendet, das Zusammenhänge zwischen verschiedenen Eigenschaften
einer Immobilie und ihrem Verkaufspreis erkennt.

Konkret wird das Modell mit Beispieldaten trainiert, die Informationen wie das durchschnittliche Einkommen, das Durchschnittsalter einer Immobilie,
die durchschnittliche Anzahl an Zimmern und Schlafzimmern sowie die Bevölkerungszahl in einer Nachbarschaft enthalten. Aus diesen Daten lernt es eine
mathematische Beziehung zwischen den Eigenschaften und dem Preis. 

____________________

**Laufende Version**

Die laufende Version des Projekts befindet sich im Jupyter Notebook „MiniMLProject.ipynb“ im folgenden Repository:
[https://github.com/gabriel-git25/MiniMLProject](https://github.com/gabriel-git25/MiniMLProject/blob/main/MiniMLProject.ipynb)

____________________

**Benutzungsanleitung (+Testhinweise)**

1. Beim Starten des Programms wird zunächst gefragt, anhand welcher Eigenschaft der Marktpreis einer Immobilie vorhergesagt werden soll.
   Dabei können Sie sich für folgende Faktoren entscheiden:
   - "Avg. Area Income" (1)
   - "Avg. Area House Age" (2)
   - "Avg. Area Number of Rooms" (3)
   - "Area Population" (4)
   - "Use all of the features above" (5)
   
   *Hinweis:*
   - Bei fehlender oder unerwarteter Eingabe wird das Programm abgebrochen, und Sie werden gebeten, erneut zu beginnen.

2. Abhängig von Ihrer Entscheidung in Schritt 1 werden Ihnen eine oder mehrere weitere Fragen gestellt.

   1. Wurde „1–4“ eingegeben, erhalten Sie die Möglichkeit, für die entsprechende Eigenschaft einen Wert anzugeben, auf dessen Grundlage
      die Vorhersage erfolgt.
   
   2. Wurde „5“ eingegeben, werden Ihnen einige weitere Fragen gestellt. Auch hier erhalten Sie bei jeder Frage die Möglichkeit, für jede Eigenschaft
      einen Wert anzugeben, auf dessen Grundlage die Vorhersage erfolgt.

   *Hinweise:*
   - Die Eingabe muss numerisch sein. Ist dies nicht der Fall, erscheint eine Fehlermeldung, und Sie müssen wieder bei Schritt 1 beginnen.
   - Fehlt lediglich die Eingabe, benutzt das Programm für die Erstellung der Vorhersage vordefinierte Werte.

3. Wurden alle vorherigen Schritte befolgt, erscheint in der Ausgabe ein Graph, der die Vorhersage visualisiert. Zusätzlich werden einige
   statistische Kennzahlen ausgegeben, die das Modell und die einzelnen Variablen beschreiben.

____________________

**Beschreibung des ML-Anteils**

Dieses Programm basiert auf dem Machine-Learning-Konzept des Supervised Learning. Konkret wird dabei ein Verfahren verwendet, das als lineare
Regression bezeichnet wird.

Ein lineares Regressionsmodell ist ein statistisches Verfahren, das Zusammenhänge zwischen einer oder mehreren Eingangsgrössen und einer Zielgrösse
beschreibt. Die Grundidee ist, dass der Immobilienpreis nicht zufällig entsteht, sondern durch mehrere messbare Faktoren beeinflusst wird.

Im einfachsten Fall basiert das Modell auf einer einzelnen erklärenden Variable (z. B. dem Durchschnittsalter der Immobilie). In dieser einfachen
linearen Regression wird nur dieser Faktor zur Preisvorhersage herangezogen. 

Sobald jedoch mehrere Eigenschaften gleichzeitig berücksichtigt werden, wird das Modell zu einer multiplen linearen Regression erweitert, was die
Vorhersage realistischer und deutlich genauer macht.

____________________

**Verwendete Tools und Libraries**

Für dieses Programm wurden folgende drei Libraries verwendet:
- "pandas" (zur Datenverarbeitung und -analyse)
- "statsmodels" (zur statistischen Modellierung, u.a Vorhersagen)
- "matplotlib" (zur Visualisierung von Daten und Ergebnissen)

*Weitere Tools:*

Zudem wurde GitHub Copilot als Programmier-Assistent eingesetzt, wodurch die Entwicklung dieses Projekts erleichtert wurde. Für die lineare 
Regression kam schliesslich eine CSV-Tabelle mit historischen Immobiliendaten zum Einsatz, die ebenfalls im Repository enthalten ist.

____________________

**Testhinweise**

Wenn nach der ersten Eingabe keine weiteren Benutzereingaben erfolgen, werden vordefinierte Werte verwendet. Diese sind nicht nur dann hilfreich, wenn keine eigenen Daten zur Verfügung stehen, sondern dienten auch der Überprüfung und dem Testen des Codes.

____________________

**Grenzen und bekannte Probleme**

*Vereinfachung der Realität*

Das Modell basiert auf einem linearen Zusammenhang zwischen den Eigenschaften einer Immobilie und dem Preis. In der Realität sind Immobilienpreise 
jedoch deutlich komplexer und werden oft durch nicht-lineare Effekte beeinflusst (z. B. Lagequalität, Nachfrage oder Infrastruktur), die durch eine 
lineare Regression nur teilweise abgebildet werden können.

*Abhängigkeit von Daten*

Die Qualität der Vorhersagen hängt stark von den verwendeten historischen Daten ab. Falls die Daten unvollständig oder verzerrt sind, können auch 
die Prognosen entsprechend ungenau sein.

____________________

**Nutzung eines Programmier-Agenten**

In diesem Projekt wurde GitHub Copilot als Programmier-Assistent zur Unterstützung eingesetzt. Dabei diente er insbesondere der Ideenfindung sowie 
als Hilfe bei der Entwicklung einzelner Codeabschnitte.

Der eigentliche Code wurde jedoch grösstenteils eigenständig erarbeitet und implementiert. Der Einsatz von Copilot beschränkte sich auf wenige, 
kleinere Situationen, in denen er auf mögliche Python-Funktionen oder -Methoden hinwies, die die Umsetzung vereinfachten. Beispiele hierfür sind 
etwa Hinweise auf Funktionen wie ".strip()" oder ".copy()".

____________________

**Quellen**

*Ursprüngliche CSV-Tabelle:*

[https://www.kaggle.com/datasets/huyngohoang/housingcsv](https://www.kaggle.com/datasets/huyngohoang/housingcsv)


   
