cmake_minimum_required(VERSION 3.10)
project(test)

find_program(BASH_PROGRAM bash)

foreach (TEST_CASE test1 test2 test3)
    add_test(
            NAME ${TEST_CASE}
            COMMAND ${BASH_PROGRAM}
            ${PROJECT_SOURCE_DIR}/test.sh
            ${BINARY_NAME}
            ${PROJECT_SOURCE_DIR}/resources/${TEST_CASE}/input.txt
            ${PROJECT_SOURCE_DIR}/resources/${TEST_CASE}/output.txt
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach (TEST_CASE)