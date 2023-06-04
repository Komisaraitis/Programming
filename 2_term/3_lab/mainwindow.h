#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void paintEvent(QPaintEvent *event);
    void mousePressEvent(QMouseEvent *event);
    QPoint na_setke(QPoint p);

private:
    Ui::MainWindow *ui;
    int HW = 20;
    QRect rect = QRect(QPoint(0,0), QSize(HW,HW));
    QPoint derevo, muravei, muha;
    int k=0;
    QVector <QLine> line;
    QVector <QPoint> pp;

};

#endif // MAINWINDOW_H
