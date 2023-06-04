#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPainter>
#include <QMouseEvent>
#include <QBrush>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    if (!p.isNull() && !q.isNull())
        painter.drawLine(p,q);
    QPoint s = q-p;
    n1 = QPoint(-s.y(), s.x());
    n2 = QPoint(s.y(), -s.x());
    painter.drawLine(q,n1+q);
    painter.drawLine(q,n2+q);
    painter.setBrush(QColor(200,120,200, 200));
    QLine CenterLine(p,q);
    QLineF CenterLineF(p.x(),p.y(),q.x(),q.y());
    CenterLineF.setAngle(CenterLineF.angle()-15);
    QPointF k=CenterLineF.p2();
    painter.drawLine(CenterLineF);
    QLineF CenterLineF1(p.x(),p.y(),q.x(),q.y());
    CenterLineF1.setAngle(CenterLineF1.angle()+15);
    QPointF k1=CenterLineF1.p2();
    painter.drawLine(CenterLineF1);


    QPointF t[] = {p, k1, k};
    painter.drawPolygon(t,3);




//    painter.setBrush(Qt::red);
//    painter.drawEllipse(q+n1,10,10);
//    painter.setBrush(Qt::green);
//    painter.drawEllipse(q+n2,10,10);



}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    count+=1;
    if (count==1)
        p = event->pos();
    if (count==2) {
        q = event->pos();


    }
    if (count>=3) {
        int c = classify(QLine(p,q), event->pos());
        qDebug("%s", c == LEFT ? "LEFT" : c == RIGHT ? "RIGHT" : "UNDEFINED" );
        int f1 = classify(QLine(q+n1, q+n2), event->pos());
        int f2 = classify(QLine(p+n2, p+n1), event->pos());
        qDebug("%s", f1 == RIGHT ? "FRONT" : f2 == RIGHT ? "BACK" : (f1&&f2) == LEFT ? "SIDE" : "UNDEFINED" );
    }

    repaint();
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{

}

int MainWindow::classify(QLineF l, QPointF p2)
{
    QPointF p0 = l.p1();
    QPointF p1 = l.p2();
    QPointF a = p1 - p0;
    QPointF b = p2 - p0;
    int g = a.x()*b.y() - b.x()*a.y();
    if (g>0)
        return RIGHT;
    if (g<0)
        return LEFT;
    return UNDEFINED;
}

