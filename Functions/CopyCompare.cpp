#include <iostream>
#include <fstream>
#include <string>
#include <vector> // Used to store lines for more flexible comparison, though not strictly necessary for line-by-line.

int main() {
    std::string file1_path = "CopiedFirst1000RowsCPP.csv";
    std::string file2_path = "CopiedFirst1000Rows.csv";

    std::ifstream file1(file1_path);
    std::ifstream file2(file2_path);

    // Check if files were opened successfully
    if (!file1.is_open()) {
        std::cerr << "Error: Could not open file: " << file1_path << std::endl;
        return 1;
    }
    if (!file2.is_open()) {
        std::cerr << "Error: Could not open file: " << file2_path << std::endl;
        file1.close(); // Close the first file if it was opened
        return 1;
    }

    std::string line1, line2;
    int line_number = 0;
    bool differences_found = false;

    std::cout << "\n--- Comparing Files ---" << std::endl;

    // Read files line by line
    while (true) {
        bool eof1 = !std::getline(file1, line1); // Read line from file1
        bool eof2 = !std::getline(file2, line2); // Read line from file2

        line_number++;

        // Both files have lines to read
        if (!eof1 && !eof2) {
            if (line1 != line2) {
                differences_found = true;
                std::cout << "\nDifference at Line " << line_number << ":" << std::endl;
                std::cout << "File 1 (" << file1_path << "): " << line1 << std::endl;
                std::cout << "File 2 (" << file2_path << "): " << line2 << std::endl;
            }
        }
        // File1 has more lines
        else if (!eof1 && eof2) {
            differences_found = true;
            std::cout << "\nDifference at Line " << line_number << ":" << std::endl;
            std::cout << "File 1 (" << file1_path << "): " << line1 << std::endl;
            std::cout << "File 2 (" << file2_path << "): <End of File>" << std::endl;
            // Print remaining lines of file1
            do {
                line_number++;
                std::cout << "\nDifference at Line " << line_number << ":" << std::endl;
                std::cout << "File 1 (" << file1_path << "): " << line1 << std::endl;
                std::cout << "File 2 (" << file2_path << "): <End of File>" << std::endl;
            } while (std::getline(file1, line1));
            break; // Exit loop as file2 has ended
        }
        // File2 has more lines
        else if (eof1 && !eof2) {
            differences_found = true;
            std::cout << "\nDifference at Line " << line_number << ":" << std::endl;
            std::cout << "File 1 (" << file1_path << "): <End of File>" << std::endl;
            std::cout << "File 2 (" << file2_path << "): " << line2 << std::endl;
            // Print remaining lines of file2
            do {
                line_number++;
                std::cout << "\nDifference at Line " << line_number << ":" << std::endl;
                std::cout << "File 1 (" << file1_path << "): <End of File>" << std::endl;
                std::cout << "File 2 (" << file2_path << "): " << line2 << std::endl;
            } while (std::getline(file2, line2));
            break; // Exit loop as file1 has ended
        }
        // Both files have reached EOF simultaneously
        else {
            break; // Both files ended
        }
    }

    // Close the files
    file1.close();
    file2.close();

    std::cout << "\n--- Comparison Finished ---" << std::endl;

    if (!differences_found) {
        std::cout << "Files are identical." << std::endl;
    } else {
        std::cout << "Differences were found between the files." << std::endl;
    }

    return 0; // Indicate successful execution
}
