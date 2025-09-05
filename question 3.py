
import turtle

def draw_edge(length, depth):
    """Recursive function to draw one edge with inward indentation pattern."""
    if depth <= 0:
        turtle.forward(length)
    else:
        length /= 3
        draw_edge(length, depth - 1.5)
        turtle.right(60)   # inward turn4
        draw_edge(length, depth - 1.5)
        turtle.left(120)   # inward bend
        draw_edge(length, depth - 1.5)
        turtle.right(60)   # return
        draw_edge(length, depth - 1.5)

def draw_polygon(sides, length, depth):
    """Draw a polygon where each edge is replaced by recursive pattern."""
    for _ in range(sides):
        draw_edge(length, depth)
        turtle.right(360 / sides)

def main():
    
    sides = int(input("Enter the number of sides: "))
    length = int(input("Enter the side length: "))
    depth = int(input("Enter the recursion depth: "))

    
    turtle.speed(0)  
    turtle.penup()
    turtle.goto(-length // 2, length // 2) 
    turtle.pendown()

 
    draw_polygon(sides, length, depth)

    turtle.done()

if __name__ == "__main__":
    main()
