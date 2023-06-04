#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}
enum{LEFT,RIGHT, UNDEFINED};//возвращает int
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void paintEvent(QPaintEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    int classify(QLineF l, QPointF p2);

private:
    Ui::MainWindow *ui;
    QPoint p,q, n1,n2;
    int count=0;




};

#endif // MAINWINDOW_H
